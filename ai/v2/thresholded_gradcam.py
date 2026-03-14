"""
thresholded_gradcam.py
======================
Thresholded Grad-CAM implementation to eliminate low-confidence noise.

Features:
- Adaptive thresholding based on attention distribution
- Morphological operations to clean noise
- Integration with attention rollout for hybrid explanations
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import zoom, binary_opening, binary_closing, label
from scipy import ndimage
import cv2


class ThresholdedGradCAM3D:
    """
    3D Grad-CAM with adaptive thresholding to remove noise and focus on
    high-confidence anatomical regions.
    """

    def __init__(self, model, target_layer):
        if not isinstance(target_layer, nn.Conv3d):
            raise ValueError(f"Target layer for Grad-CAM must be nn.Conv3d, got: {type(target_layer)}")
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor,
        target_class=None,
        threshold_method="adaptive",
        threshold_percentile=75,
        morphological_cleaning=True
    ):
        """
        Generate thresholded Grad-CAM heatmap.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input volume [B, 1, D, H, W]
        target_class : int, optional
            Target class for explanation
        threshold_method : str
            "adaptive", "percentile", or "otsu"
        threshold_percentile : float
            Percentile for thresholding (if method="percentile")
        morphological_cleaning : bool
            Whether to apply morphological operations

        Returns
        -------
        tuple
            (thresholded_cam, raw_cam) - both [D, H, W]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        # Compute raw Grad-CAM
        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)
        raw_cam = torch.sum(weights * activations, dim=1).squeeze()
        raw_cam = torch.relu(raw_cam)

        # Convert to numpy
        raw_cam_np = raw_cam.cpu().detach().numpy()

        # Upsample to input resolution
        input_shape = input_tensor.shape[2:]  # [D, H, W]
        cam_shape = raw_cam_np.shape

        if cam_shape != input_shape:
            scale_factors = np.array(input_shape) / np.array(cam_shape)
            raw_cam_np = zoom(raw_cam_np, scale_factors, order=1)

        # Apply thresholding
        thresholded_cam = self._apply_thresholding(
            raw_cam_np, method=threshold_method, percentile=threshold_percentile
        )

        # Morphological cleaning
        if morphological_cleaning:
            thresholded_cam = self._morphological_cleaning(thresholded_cam)

        # Renormalize
        if thresholded_cam.max() > thresholded_cam.min():
            thresholded_cam = (thresholded_cam - thresholded_cam.min()) / (thresholded_cam.max() - thresholded_cam.min())

        return thresholded_cam, raw_cam_np

    def _apply_thresholding(self, cam, method="adaptive", percentile=75):
        """
        Apply thresholding to remove low-confidence regions.
        """
        if method == "adaptive":
            # Adaptive thresholding based on local statistics
            from skimage.filters import threshold_local
            block_size = min(cam.shape) // 8 * 2 + 1  # Odd block size
            threshold = threshold_local(cam, block_size, offset=0)
            return np.where(cam > threshold, cam, 0)

        elif method == "percentile":
            # Percentile-based global thresholding
            threshold = np.percentile(cam[cam > 0], percentile) if np.any(cam > 0) else 0
            return np.where(cam > threshold, cam, 0)

        elif method == "otsu":
            # Otsu's method
            from skimage.filters import threshold_otsu
            try:
                threshold = threshold_otsu(cam[cam > 0]) if np.any(cam > 0) else 0
                return np.where(cam > threshold, cam, 0)
            except:
                # Fallback to percentile
                return self._apply_thresholding(cam, "percentile", percentile)

        else:
            return cam

    def _morphological_cleaning(self, cam, min_size=100):
        """
        Apply morphological operations to clean noise.
        """
        # Binarize
        binary = cam > 0

        # Remove small connected components
        labeled, num_features = label(binary)
        if num_features > 1:
            component_sizes = np.bincount(labeled.ravel())
            component_sizes[0] = 0  # Ignore background

            # Keep only components above minimum size
            for i in range(1, num_features + 1):
                if component_sizes[i] < min_size:
                    binary[labeled == i] = False

        # Apply opening to remove noise
        binary = binary_opening(binary, iterations=1)

        # Apply closing to fill small gaps
        binary = binary_closing(binary, iterations=1)

        # Apply back to original values
        cleaned_cam = cam * binary.astype(float)

        return cleaned_cam


def combine_thresholded_gradcam_and_rollout(
    thresholded_gradcam,
    attention_rollout,
    combination_method="multiply",
    alpha=0.5
):
    """
    Combine thresholded Grad-CAM with attention rollout.

    Parameters
    ----------
    thresholded_gradcam : np.ndarray
        Thresholded Grad-CAM [D, H, W]
    attention_rollout : np.ndarray
        Attention rollout [D, H, W]
    combination_method : str
        "multiply", "add", "max", or "weighted"
    alpha : float
        Weight for weighted combination

    Returns
    -------
    np.ndarray
        Combined explanation [D, H, W]
    """
    # Ensure same shape
    if thresholded_gradcam.shape != attention_rollout.shape:
        # Resize rollout to match gradcam
        attention_rollout = zoom(
            attention_rollout,
            np.array(thresholded_gradcam.shape) / np.array(attention_rollout.shape),
            order=1
        )

    if combination_method == "multiply":
        combined = thresholded_gradcam * attention_rollout
    elif combination_method == "add":
        combined = thresholded_gradcam + attention_rollout
    elif combination_method == "max":
        combined = np.maximum(thresholded_gradcam, attention_rollout)
    elif combination_method == "weighted":
        combined = alpha * thresholded_gradcam + (1 - alpha) * attention_rollout
    else:
        combined = thresholded_gradcam

    # Renormalize
    if combined.max() > combined.min():
        combined = (combined - combined.min()) / (combined.max() - combined.min())

    return combined


def gradcam_confidence_score(gradcam, ground_truth_mask=None):
    """
    Compute confidence score for Grad-CAM explanation.

    Parameters
    ----------
    gradcam : np.ndarray
        Grad-CAM heatmap [D, H, W]
    ground_truth_mask : np.ndarray, optional
        Binary mask of expected anatomical regions

    Returns
    -------
    dict
        Confidence metrics
    """
    metrics = {}

    # Basic statistics
    metrics['max_value'] = gradcam.max()
    metrics['mean_value'] = gradcam.mean()
    metrics['std_value'] = gradcam.std()
    metrics['sparsity'] = np.mean(gradcam == 0)

    # Focus score (concentration of high values)
    threshold = np.percentile(gradcam, 90)
    high_attention_mask = gradcam > threshold
    metrics['focus_score'] = np.sum(high_attention_mask) / np.prod(gradcam.shape)

    # Anatomical alignment (if ground truth provided)
    if ground_truth_mask is not None:
        intersection = np.sum(high_attention_mask & ground_truth_mask)
        union = np.sum(high_attention_mask | ground_truth_mask)
        metrics['dice_score'] = 2 * intersection / (np.sum(high_attention_mask) + np.sum(ground_truth_mask)) if (np.sum(high_attention_mask) + np.sum(ground_truth_mask)) > 0 else 0
        metrics['jaccard_score'] = intersection / union if union > 0 else 0

    return metrics
