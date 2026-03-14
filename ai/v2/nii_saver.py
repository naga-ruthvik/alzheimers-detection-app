"""v2/nii_saver.py
Clean, medical-standard utilities to save heatmaps and 2D overlay PNGs.

Provides:
- save_heatmap_as_nifti(heatmap, reference_nifti_path, output_path)
- create_overlay_nifti(original_nifti_path, heatmap, output_path)
- save_2d_slices_with_overlay(original_nifti_path, overlay, output_path)

Requirements / behavior:
- Strict orientation: Axial (Z), Coronal (Y), Sagittal (X)
- Anatomical smoothing: gaussian_filter(sigma=1.5)
- Background masking: threshold original MRI to remove background
- Color mapping: 'jet' default, alpha overlay default 0.4
"""

from typing import Tuple
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def save_heatmap_as_nifti(heatmap: np.ndarray, reference_nifti_path: str, output_path: str):
    """Save a 3D scalar heatmap as a NIfTI matching the reference affine and shape.

    If heatmap shape differs, it will be resampled to reference shape using linear interpolation.
    """
    ref = nib.load(reference_nifti_path)
    ref_data = ref.get_fdata()
    target_shape = ref_data.shape

    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.shape != target_shape:
        factors = [t / s for t, s in zip(target_shape, heatmap.shape)]
        heatmap = zoom(heatmap, factors, order=1)

    # Normalize to 0..1 for storage (keeps floating precision)
    hmin, hmax = float(np.nanmin(heatmap)), float(np.nanmax(heatmap))
    if hmax - hmin > 0:
        heatmap_norm = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap_norm = np.zeros_like(heatmap)

    img = nib.Nifti1Image(heatmap_norm.astype(np.float32), ref.affine)
    img.header.set_data_dtype(np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(img, output_path)


def create_overlay_nifti(original_nifti_path: str, heatmap: np.ndarray, output_path: str, alpha: float = 0.6, colormap: str = "jet"):
    """Create an RGB(A) overlay NIfTI by combining original anatomy and colored heatmap.

    The function will resample the heatmap to the original shape if needed, normalize,
    apply a colormap, and save an RGBA (4D) NIfTI where the last dimension is R,G,B,A.
    """
    orig = nib.load(original_nifti_path)
    orig_data = orig.get_fdata()
    target_shape = orig_data.shape

    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.shape != target_shape:
        factors = [t / s for t, s in zip(target_shape, heatmap.shape)]
        heatmap = zoom(heatmap, factors, order=1)

    # Smooth and mask background
    heat_smooth = gaussian_filter(heatmap, sigma=1.5)
    orig_norm = (orig_data - np.nanmin(orig_data)) / (np.nanmax(orig_data) - np.nanmin(orig_data) + 1e-8)
    brain_mask = (orig_norm > 0.05).astype(np.float32)
    heat_smooth *= brain_mask

    # Normalize heatmap 0..1
    hmin, hmax = float(np.nanmin(heat_smooth)), float(np.nanmax(heat_smooth))
    if hmax - hmin > 0:
        heat_norm = (heat_smooth - hmin) / (hmax - hmin)
    else:
        heat_norm = np.zeros_like(heat_smooth)

    # Apply colormap slice-wise to build RGBA volume
    cmap = cm.get_cmap(colormap)
    rgba_stack = np.zeros((*heat_norm.shape, 4), dtype=np.float32)
    # iterate z for memory / clarity
    for z in range(heat_norm.shape[0]):
        colored = cmap(heat_norm[z, :, :])  # H x W x 4
        rgba_stack[z] = colored

    # Inject alpha from heatmap scaled by provided alpha and brain mask
    rgba_stack[..., 3] = rgba_stack[..., 3] * alpha * brain_mask

    # Save as 4D NIfTI (Z,Y,X,4)
    img = nib.Nifti1Image(rgba_stack.astype(np.float32), orig.affine)
    img.header.set_data_dtype(np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(img, output_path)


def save_2d_slices_with_overlay(original_nifti_path: str, overlay: np.ndarray, output_path: str,
                                slices_per_view: int = 3, figsize: Tuple[int, int] = (12, 10),
                                overlay_text: str = None, colormap: str = "jet", alpha: float = 0.4):
    """Save a PNG grid of central axial/coronal/sagittal slices with smoothed, masked overlay.

    Orientation conventions (strict):
    - Axial: slices along Z dimension -> `orig[z, :, :]` (top row)
    - Coronal: slices along Y dimension -> `orig[:, y, :]` (middle row)
    - Sagittal: slices along X dimension -> `orig[:, :, x]` (bottom row)

    The overlay will be smoothed (gaussian, sigma=1.5) and masked with a brain mask
    derived from the original MRI (thresholding). Colormap uses `colormap` with transparency `alpha`.
    """
    orig_nib = nib.load(original_nifti_path)
    orig_data = orig_nib.get_fdata()
    if orig_data.ndim == 4:
        orig_data = orig_data[..., 0]

    orig = np.asarray(orig_data, dtype=np.float32)
    orig_norm = (orig - np.nanmin(orig)) / (np.nanmax(orig) - np.nanmin(orig) + 1e-8)

    overlay = np.asarray(overlay, dtype=np.float32)
    # Resample overlay to match original shape if needed
    if overlay.shape != orig.shape:
        factors = [o / s for o, s in zip(orig.shape, overlay.shape)] if len(overlay.shape) == 3 else [o / s for o, s in zip(orig.shape, overlay.shape[:3])]
        overlay = zoom(overlay, factors, order=1)

    # Smooth overlay to remove blocky artifacts
    overlay_sm = gaussian_filter(overlay, sigma=1.5)

    # Brain mask from original
    brain_mask = (orig_norm > 0.05).astype(np.float32)
    overlay_masked = overlay_sm * brain_mask

    # Normalize overlay
    lo, hi = float(np.nanmin(overlay_masked)), float(np.nanmax(overlay_masked))
    if hi - lo > 0:
        overlay_norm = (overlay_masked - lo) / (hi - lo)
    else:
        overlay_norm = np.zeros_like(overlay_masked)

    D, H, W = orig.shape

    def central_slices(dim, n):
        if n == 1:
            return [dim // 2]
        step = max(1, (dim - 1) // (n + 1))
        return [step * (i + 1) for i in range(n)]

    axial_slices = central_slices(D, slices_per_view)
    coronal_slices = central_slices(H, slices_per_view)
    sagittal_slices = central_slices(W, slices_per_view)

    fig, axes = plt.subplots(3, slices_per_view, figsize=figsize)
    title = os.path.basename(output_path).replace("_slices.png", "")
    if overlay_text:
        title = f"{title} - {overlay_text}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    cmap = cm.get_cmap(colormap)

    # Row 1: Axial (Z) - looking down from top; show ventricles butterfly
    for i, z in enumerate(axial_slices):
        img = np.transpose(orig[z, :, :], (1, 0))
        heat = np.transpose(overlay_norm[z, :, :], (1, 0))
        axes[0, i].imshow(img, cmap='gray', aspect='equal')
        axes[0, i].imshow(heat, cmap=colormap, alpha=alpha, aspect='equal')
        axes[0, i].set_title(f'Axial {z}')
        axes[0, i].axis('off')

    # Row 2: Coronal (Y) - looking from front
    for i, y in enumerate(coronal_slices):
        img = np.transpose(orig[:, y, :], (1, 0))
        heat = np.transpose(overlay_norm[:, y, :], (1, 0))
        axes[1, i].imshow(img, cmap='gray', aspect='equal')
        axes[1, i].imshow(heat, cmap=colormap, alpha=alpha, aspect='equal')
        axes[1, i].set_title(f'Coronal {y}')
        axes[1, i].axis('off')

    # Row 3: Sagittal (X) - looking from side
    for i, x in enumerate(sagittal_slices):
        img = np.transpose(orig[:, :, x], (1, 0))
        heat = np.transpose(overlay_norm[:, :, x], (1, 0))
        axes[2, i].imshow(img, cmap='gray', aspect='equal')
        axes[2, i].imshow(heat, cmap=colormap, alpha=alpha, aspect='equal')
        axes[2, i].set_title(f'Sagittal {x}')
        axes[2, i].axis('off')

    # Colorbar (use a ScalarMappable)
    sm = cm.ScalarMappable(cmap=colormap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Attention/Importance', rotation=270, labelpad=15)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
