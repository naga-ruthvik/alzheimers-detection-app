"""
attention_rollout_hybrid.py
===========================
Attention Rollout implementation for HCCT hybrid CNN-Transformer architecture.

Features:
- Projects transformer tokens back to 3D spatial volume
- Handles hybrid CNN-Transformer attention flow
- Combines rollout with spatial projection
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import zoom
from einops import rearrange
import math


def compute_attention_rollout_hybrid(
    model,
    input_tensor,
    head_fusion="mean",
    discard_ratio=0.9
):
    """
    Compute Attention Rollout for HCCT hybrid architecture.

    Projects transformer attention back to spatial volume through learned
    patch embeddings and CNN feature maps.

    Parameters
    ----------
    model : HCCTModel
        Trained HCCT model with attention weight capture
    input_tensor : torch.Tensor
        Input volume [B, 1, D, H, W]
    head_fusion : str
        How to fuse attention heads ("mean", "max", "min")
    discard_ratio : float
        Ratio of lowest-attention tokens to discard

    Returns
    -------
    np.ndarray
        3D attention heatmap [D, H, W]
    """
    model.eval()
    device = input_tensor.device
    B, C, D, H, W = input_tensor.shape

    # Forward pass to capture attention weights
    with torch.no_grad():
        _ = model(input_tensor)

    attention_weights = model.last_attention_weights  # List of [B, H, N, N]

    if not attention_weights:
        raise ValueError("No attention weights captured. Use TransformerEncoderWithAttn.")

    # Number of patches (excluding CLS token)
    num_patches = attention_weights[0].shape[-1] - 1
    patch_side = int(round(num_patches ** (1/3)))  # Assuming cubic patches

    # Initialize rollout matrix (identity for 0 layers)
    N = attention_weights[0].shape[-1]  # Includes CLS token
    rollout = torch.eye(N, device=device)

    # Accumulate attention across layers
    for layer_attn in attention_weights:
        # layer_attn: [B, num_heads, N, N]
        attn = layer_attn[0]  # [num_heads, N, N]

        # Fuse heads
        if head_fusion == "mean":
            attn = attn.mean(dim=0)
        elif head_fusion == "max":
            attn = attn.max(dim=0)[0]
        elif head_fusion == "min":
            attn = attn.min(dim=0)[0]

        # Add residual connection (skip connections in transformer)
        identity = torch.eye(N, device=device)
        attn = 0.5 * attn + 0.5 * identity

        # Row-normalize
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Accumulate
        rollout = attn @ rollout

    # Extract CLS token attention to patches
    cls_attention = rollout[0, 1:]  # [num_patches]

    # Discard lowest attention tokens (focus on important regions)
    if discard_ratio > 0:
        threshold = torch.quantile(cls_attention, discard_ratio)
        cls_attention = torch.where(cls_attention > threshold, cls_attention, torch.zeros_like(cls_attention))

    # Reshape to 3D patch grid
    patch_grid = cls_attention.view(patch_side, patch_side, patch_side).cpu().numpy()

    # Upsample to full volume resolution
    scale_factor = D / patch_side
    attention_volume = zoom(patch_grid, (scale_factor, scale_factor, scale_factor), order=1)

    # Normalize to [0, 1]
    if attention_volume.max() > attention_volume.min():
        attention_volume = (attention_volume - attention_volume.min()) / (attention_volume.max() - attention_volume.min())

    return attention_volume


def project_tokens_to_volume(
    model,
    input_tensor,
    token_attention,
    method="bilinear"
):
    """
    Project transformer token attentions back to 3D spatial volume.

    Parameters
    ----------
    model : HCCTModel
        The model (for accessing patch embeddings if needed)
    input_tensor : torch.Tensor
        Original input volume [B, 1, D, H, W]
    token_attention : torch.Tensor
        Attention weights for each token [num_patches]
    method : str
        Interpolation method ("nearest", "bilinear", "bicubic")

    Returns
    -------
    np.ndarray
        Spatial attention volume [D, H, W]
    """
    B, C, D, H, W = input_tensor.shape
    num_patches = token_attention.shape[0]
    patch_side = int(round(num_patches ** (1/3)))

    # Reshape token attention to patch grid
    patch_attention = token_attention.view(patch_side, patch_side, patch_side)

    # Convert to torch tensor for interpolation
    patch_attention = patch_attention.unsqueeze(0).unsqueeze(0)  # [1, 1, patch_side, patch_side, patch_side]

    # Upsample to full resolution
    upsampled = torch.nn.functional.interpolate(
        patch_attention,
        size=(D, H, W),
        mode=method,
        align_corners=False if method != "nearest" else None
    )

    attention_volume = upsampled.squeeze().cpu().numpy()

    # Normalize
    if attention_volume.max() > attention_volume.min():
        attention_volume = (attention_volume - attention_volume.min()) / (attention_volume.max() - attention_volume.min())

    return attention_volume


def compute_hybrid_attention_flow(
    model,
    input_tensor,
    gradcam_weights=None,
    alpha=0.5
):
    """
    Compute hybrid attention flow combining transformer rollout and CNN gradients.

    Parameters
    ----------
    model : HCCTModel
        Trained model
    input_tensor : torch.Tensor
        Input volume [B, 1, D, H, W]
    gradcam_weights : np.ndarray, optional
        Grad-CAM weights from CNN layers
    alpha : float
        Weight for combining transformer and CNN attentions

    Returns
    -------
    dict
        Dictionary with different attention maps
    """
    # Get transformer attention rollout
    transformer_attention = compute_attention_rollout_hybrid(model, input_tensor)

    results = {
        "transformer_rollout": transformer_attention,
    }

    # If Grad-CAM weights provided, create hybrid
    if gradcam_weights is not None:
        # Ensure same shape
        if gradcam_weights.shape != transformer_attention.shape:
            gradcam_resized = zoom(
                gradcam_weights,
                np.array(transformer_attention.shape) / np.array(gradcam_weights.shape),
                order=1
            )
        else:
            gradcam_resized = gradcam_weights

        # Combine attentions
        hybrid_attention = alpha * transformer_attention + (1 - alpha) * gradcam_resized

        # Renormalize
        if hybrid_attention.max() > hybrid_attention.min():
            hybrid_attention = (hybrid_attention - hybrid_attention.min()) / (hybrid_attention.max() - hybrid_attention.min())

        results["hybrid_attention"] = hybrid_attention
        results["cnn_gradcam"] = gradcam_resized

    return results


def rollout_to_nifti(attention_volume, reference_nifti_path, output_path):
    """
    Save attention rollout as NIfTI file for 3D visualization.

    Parameters
    ----------
    attention_volume : np.ndarray
        3D attention map [D, H, W]
    reference_nifti_path : str
        Path to original NIfTI for header/affine
    output_path : str
        Output NIfTI path
    """
    import nibabel as nib

    # Load reference
    ref_img = nib.load(reference_nifti_path)

    # Resize attention to match reference shape if needed
    if attention_volume.shape != ref_img.shape:
        attention_volume = zoom(
            attention_volume,
            np.array(ref_img.shape) / np.array(attention_volume.shape),
            order=1
        )

    # Create new NIfTI
    attention_nifti = nib.Nifti1Image(attention_volume, ref_img.affine, ref_img.header)
    nib.save(attention_nifti, output_path)
