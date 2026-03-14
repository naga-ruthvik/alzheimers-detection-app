"""
enhanced_preprocessing.py
=========================
Enhanced preprocessing pipeline for anatomical grounding in Alzheimer's detection.

Key improvements:
- Supratentorial brain isolation with advanced skull stripping
- 3D Random Affine transformations to break edge artifacts
- Anatomical region masking for adversarial training
- Integration with existing OASIS preprocessing
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import (
    zoom, binary_fill_holes, binary_dilation, binary_erosion,
    affine_transform, rotate, shift
)
from scipy import ndimage
import nibabel as nib
import random
import math


def enhanced_skull_strip(img, margin=5, keep_supratentorial=True):
    """
    Fast skull stripping optimized for training speed.

    Simplified version of advanced skull stripping for better performance.
    """
    # Single threshold approach (faster)
    threshold = np.percentile(img[img > 0], 30) if img.max() > 0 else img.mean()
    mask = img > threshold

    # Fill holes
    mask = binary_fill_holes(mask)

    # Dilate to recover border voxels
    mask = binary_dilation(mask, iterations=2)

    # Apply brain mask
    img = img * mask.astype(np.float32)

    # Supratentorial isolation (if requested) - simplified
    if keep_supratentorial:
        D = img.shape[0]
        cutoff = int(D * 0.18)  # Zero inferior 18% (cerebellum)
        img[:cutoff, :, :] = 0

    # Tight bounding box with margin
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return img

    min_c = np.maximum(coords.min(axis=0) - margin, 0)
    max_c = np.minimum(coords.max(axis=0) + margin + 1, np.array(img.shape))
    return img[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]


def isolate_supratentorial(img, tentorium_frac=0.25):
    """
    Mask out infratentorial regions below the tentorium cerebelli.

    Parameters
    ----------
    img : np.ndarray
        3D volume [D, H, W]
    tentorium_frac : float
        Fraction of superior-inferior axis where tentorium is located

    Returns
    -------
    np.ndarray
        Volume with infratentorial regions zeroed
    """
    D, H, W = img.shape
    # Tentorium typically at ~25% from inferior edge in standard orientations
    tentorium_z = int(D * tentorium_frac)

    # Zero inferior portion (cerebellum, brainstem)
    img[:tentorium_z, :, :] = 0
    return img


def random_3d_affine_transform(img, rotation_range=15, shift_range=0.1, scale_range=0.1):
    """
    Apply random 3D affine transformation to break reliance on fixed scan artifacts.

    Parameters
    ----------
    img : np.ndarray
        Input 3D volume [D, H, W]
    rotation_range : float
        Max rotation angle in degrees
    shift_range : float
        Max shift as fraction of image size
    scale_range : float
        Max scale variation

    Returns
    -------
    np.ndarray
        Transformed volume
    """
    D, H, W = img.shape

    # Random rotation angles
    angles = [
        random.uniform(-rotation_range, rotation_range),
        random.uniform(-rotation_range, rotation_range),
        random.uniform(-rotation_range, rotation_range)
    ]

    # Random shifts
    shifts = [
        random.uniform(-shift_range * D, shift_range * D),
        random.uniform(-shift_range * H, shift_range * H),
        random.uniform(-shift_range * W, shift_range * W)
    ]

    # Random scaling
    scales = [
        random.uniform(1 - scale_range, 1 + scale_range),
        random.uniform(1 - scale_range, 1 + scale_range),
        random.uniform(1 - scale_range, 1 + scale_range)
    ]

    # Apply transformations sequentially
    transformed = img.copy()

    # Rotation
    for axis, angle in enumerate(angles):
        if abs(angle) > 1e-3:  # Only rotate if significant
            transformed = rotate(transformed, angle, axes=(axis, (axis+1)%3),
                               reshape=False, mode='constant', cval=0)

    # Scaling and shifting via affine transform
    scale_matrix = np.diag(scales + [1])  # 4x4 matrix
    shift_matrix = np.eye(4)
    shift_matrix[:3, 3] = shifts

    affine_matrix = shift_matrix @ scale_matrix

    transformed = affine_transform(
        transformed, affine_matrix[:3, :3], offset=shifts,
        mode='constant', cval=0
    )

    return transformed


def anatomical_region_mask(img, region='medial_temporal', mask_prob=0.3):
    """
    Create anatomical region masks for adversarial training.

    Parameters
    ----------
    img : np.ndarray
        3D volume [D, H, W]
    region : str
        Anatomical region to mask ('medial_temporal', 'hippocampus', 'ventricles')
    mask_prob : float
        Probability of applying masking

    Returns
    -------
    np.ndarray
        Volume with specified region randomly masked
    """
    if random.random() > mask_prob:
        return img

    D, H, W = img.shape
    masked_img = img.copy()

    if region == 'medial_temporal':
        # Medial temporal lobe: posterior 1/3, medial 1/3
        z_start, z_end = int(0.4 * D), D
        y_start, y_end = int(0.3 * H), int(0.7 * H)
        x_start, x_end = int(0.3 * W), int(0.7 * W)

    elif region == 'hippocampus':
        # Hippocampus region approximation
        z_start, z_end = int(0.45 * D), int(0.65 * D)
        y_start, y_end = int(0.4 * H), int(0.6 * H)
        x_start, x_end = int(0.4 * W), int(0.6 * W)

    elif region == 'ventricles':
        # Ventricular system approximation
        z_start, z_end = int(0.3 * D), int(0.7 * D)
        y_start, y_end = int(0.45 * H), int(0.55 * H)
        x_start, x_end = int(0.45 * W), int(0.55 * W)

    else:
        return img

    # Random box within the region
    box_size = (
        random.randint(int(0.1*D), int(0.3*D)),
        random.randint(int(0.1*H), int(0.3*H)),
        random.randint(int(0.1*W), int(0.3*W))
    )

    z0 = random.randint(z_start, max(z_start, z_end - box_size[0]))
    y0 = random.randint(y_start, max(y_start, y_end - box_size[1]))
    x0 = random.randint(x_start, max(x_start, x_end - box_size[2]))

    masked_img[z0:z0+box_size[0], y0:y0+box_size[1], x0:x0+box_size[2]] = 0

    return masked_img


def enhanced_preprocessing_pipeline(img_path, img_size=128, augment=False):
    """
    Fast preprocessing pipeline based on working original code.

    Parameters
    ----------
    img_path : str
        Path to NIfTI file
    img_size : int
        Target volume size
    augment : bool
        Whether to apply data augmentation

    Returns
    -------
    torch.Tensor
        Preprocessed volume [1, 1, D, H, W]
    """
    # Load NIfTI
    img_nii = nib.load(img_path)
    img = img_nii.get_fdata()
    if len(img.shape) > 3:
        img = np.squeeze(img)
    img = img.astype(np.float32)

    # Fast skull stripping (from original)
    threshold = np.percentile(img, 30)
    mask = img > threshold
    mask = binary_fill_holes(mask)
    mask = binary_dilation(mask, iterations=2)
    img = img * mask.astype(np.float32)

    # Tight crop
    coords = np.argwhere(mask)
    if len(coords) > 0:
        min_c = np.maximum(coords.min(axis=0) - 5, 0)
        max_c = np.minimum(coords.max(axis=0) + 6, np.array(img.shape))
        img = img[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]

    # Robust normalization
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip(img, p1, p99)
    mean, std = img.mean(), img.std()
    if std > 0:
        img = (img - mean) / std

    # Resize
    current_shape = img.shape
    factors = (img_size / current_shape[0], img_size / current_shape[1], img_size / current_shape[2])
    img = zoom(img, factors, order=1)

    # Zero borders
    border = 8
    img[:border, :, :] = 0
    img[-border:, :, :] = 0
    img[:, :border, :] = 0
    img[:, -border:, :] = 0
    img[:, :, :border] = 0
    img[:, :, -border:] = 0

    # Zero cerebellum
    cutoff = int(img.shape[0] * 0.18)
    img[:cutoff, :, :] = 0

    # Data augmentation
    if augment:
        if random.random() > 0.5:
            img = np.flip(img, axis=0).copy()
        if random.random() > 0.5:
            img = np.flip(img, axis=1).copy()
        noise_level = random.uniform(0.0, 0.02)
        img = img + np.random.normal(0, noise_level, img.shape).astype(np.float32)
        # Adversarial erase
        if random.random() > 0.5:
            D, H, W = img.shape
            z_max = int(D * 0.18)
            if z_max >= 4:
                z_size = random.randint(4, min(z_max, 16))
                y_size = random.randint(H // 4, H)
                x_size = random.randint(W // 4, W)
                z0 = random.randint(0, max(0, z_max - z_size))
                y0 = random.randint(0, H - y_size)
                x0 = random.randint(0, W - x_size)
                img[z0:z0+z_size, y0:y0+y_size, x0:x0+x_size] = 0.0

    # Convert to tensor
    img = torch.from_numpy(img.copy()).unsqueeze(0)
    return img