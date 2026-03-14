"""
explainability_v2.py
====================
NEW explainability module based on:
  • Attention Rollout  (Abnar & Zuidema 2020)
  • Hybrid Grad-CAM   (Rollout × raw Grad-CAM)

This file imports HCCTModel from the updated train_oasis.py (which exposes
attention weights via `model.last_attention_weights`). The existing
GradCAM3D in test_and_visualize.py is kept intact; this file adds on top.

Usage
-----
python explainability_v2.py --checkpoint models/best_model.pth
                             --csv       data/oasis1_nifti/dataset.csv
                             [--out_dir  explainability_v2_results]
                             [--num_subjects 5]

Output per subject
------------------
  <out_dir>/<subject_id>_v2_explainability.png
     Row 1 — Original MRI  (axial, coronal, sagittal)
     Row 2 — Attention Rollout heatmap
     Row 3 — Grad-CAM (raw)
     Row 4 — Hybrid CAM  (Rollout × Grad-CAM)   ← main output

Requirements
------------
  pip install torch nibabel numpy scipy matplotlib einops scikit-learn pandas tqdm
"""

import os
import sys
import argparse
import math
import random

import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import zoom

# ---------------------------------------------------------------------------
# Import the updated model + preprocessing from train_oasis.py
# ---------------------------------------------------------------------------
# Make sure train_oasis.py is on the Python path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from train_oasis import (
    HCCTModel,
    skull_strip,
    robust_normalize,
    zero_border,
    zero_cerebellum_border,
    CONFIG,
)

# Default config; callers can override via CLI
DEVICE      = CONFIG["device"]
IMAGE_SIZE  = CONFIG["image_size"]


# ===========================================================================
# 1.  PREPROCESSING  (mirrors OASISDataset.__getitem__ exactly)
# ===========================================================================

def preprocess_volume(img_path: str, img_size: int = IMAGE_SIZE) -> torch.Tensor:
    """
    Load a NIfTI volume, apply the full preprocessing pipeline, and return a
    [1, 1, D, H, W] float32 tensor ready for the model.
    """
    img_nii = nib.load(img_path)
    img = img_nii.get_fdata()
    if len(img.shape) > 3:
        img = np.squeeze(img)
    img = img.astype("float32")

    img = skull_strip(img)
    img = robust_normalize(img)

    factors = (img_size / img.shape[0],
               img_size / img.shape[1],
               img_size / img.shape[2])
    img = zoom(img, factors, order=1)

    img = zero_border(img, border=8)
    img = zero_cerebellum_border(img, inferior_frac=0.18)

    tensor = torch.from_numpy(img.copy()).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    return tensor.float()


# ===========================================================================
# 2.  ATTENTION ROLLOUT  (Abnar & Zuidema, 2020)
# ===========================================================================

def compute_attention_rollout(
    model: nn.Module,
    input_tensor: torch.Tensor,
    head_fusion: str = "mean",
) -> np.ndarray:
    """
    Compute Attention Rollout for the given volume and return a 3-D heatmap
    at the same spatial resolution as the CNN output (8×8×8), then upsample
    to (IMG_SIZE, IMG_SIZE, IMG_SIZE) for overlay.

    Parameters
    ----------
    model        : HCCTModel (must have already run a forward pass, OR we run
                   one inside this function)
    input_tensor : [1, 1, D, H, W]  preprocessed volume
    head_fusion  : one of {"mean","max","min"} — how to collapse heads before
                   accumulating rollout

    Returns
    -------
    np.ndarray of shape (IMG_SIZE, IMG_SIZE, IMG_SIZE) — values in [0, 1]
    after min-max normalisation.
    """
    model.eval()
    input_tensor = input_tensor.to(DEVICE)

    # Run a forward pass to populate model.last_attention_weights
    with torch.no_grad():
        _ = model(input_tensor)

    attn_weights = model.last_attention_weights   # list of [B, H, N, N]
    if len(attn_weights) == 0:
        raise RuntimeError(
            "model.last_attention_weights is empty. "
            "Make sure HCCTModel from train_oasis.py is used (TransformerEncoderWithAttn)."
        )

    # --- Rollout algorithm ---
    # N = num_patches + 1 (CLS token at index 0)
    N = attn_weights[0].shape[-1]
    rollout = torch.eye(N, device=DEVICE)   # identity = "0-layer rollout"

    for layer_attn in attn_weights:
        # layer_attn: [B, num_heads, N, N] (B=1 here)
        A = layer_attn[0]          # [num_heads, N, N]

        if head_fusion == "mean":
            A = A.mean(dim=0)      # [N, N]
        elif head_fusion == "max":
            A = A.max(dim=0).values
        elif head_fusion == "min":
            A = A.min(dim=0).values
        else:
            raise ValueError(f"Unknown head_fusion: {head_fusion}")

        # Add residual identity (attention can propagate through skip connections)
        A = 0.5 * A + 0.5 * torch.eye(N, device=DEVICE)

        # Row-normalise to keep probabilities summing to 1
        A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Accumulate
        rollout = A @ rollout

    # Extract CLS → patch attention row, drop CLS self-attention
    cls_attn = rollout[0, 1:]    # [N-1] = [num_patches]

    # Reshape back to CNN spatial output: 8×8×8
    side = int(round(cls_attn.numel() ** (1.0 / 3)))
    if side ** 3 != cls_attn.numel():
        raise RuntimeError(
            f"Cannot reshape {cls_attn.numel()} patch tokens into a cube. "
            f"Expected num_patches to be a perfect cube (got {cls_attn.numel()})."
        )

    rollout_3d = cls_attn.cpu().numpy().reshape(side, side, side)

    # Upsample to full image size
    scale = IMAGE_SIZE / side
    rollout_full = zoom(rollout_3d, (scale, scale, scale), order=1)

    # Min-max normalise to [0, 1]
    vmin, vmax = rollout_full.min(), rollout_full.max()
    if vmax > vmin:
        rollout_full = (rollout_full - vmin) / (vmax - vmin)
    else:
        rollout_full = np.zeros_like(rollout_full)

    return rollout_full   # (128, 128, 128)


# ===========================================================================
# 3.  GRAD-CAM (raw, applied to the final CNN block output)
# ===========================================================================

class GradCAM3D_v2:
    """
    A minimal 3-D Grad-CAM implementation for the HCCT CNN backbone.

    Target layer: the last ConvBlock inside model.cnn  (index -1),
    i.e. the output is shape [B, 256, 8, 8, 8].
    """

    def __init__(self, model: nn.Module):
        self.model     = model
        self._feats    = None
        self._grads    = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        # Last ConvBlock is index -1 inside model.cnn (a nn.Sequential)
        target_layer = list(self.model.cnn.children())[-1]   # ConvBlock(128→256)

        def fwd_hook(module, inp, out):
            self._feats = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._grads = grad_out[0].detach()

        self._hook_handles.append(target_layer.register_forward_hook(fwd_hook))
        self._hook_handles.append(target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Returns Grad-CAM heatmap of shape (IMG_SIZE, IMG_SIZE, IMG_SIZE).
        """
        self.model.eval()
        input_tensor = input_tensor.to(DEVICE).requires_grad_(True)

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients over spatial dims [D, H, W]
        weights = self._grads.mean(dim=(2, 3, 4), keepdim=True)  # [B,256,1,1,1]
        cam     = (weights * self._feats).sum(dim=1).squeeze(0)   # [D, H, W]=8³
        cam     = torch.clamp(cam, min=0).cpu().numpy()

        # Upsample
        scale = IMAGE_SIZE / cam.shape[0]
        cam   = zoom(cam, (scale, scale, scale), order=1)

        # Normalise
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


# ===========================================================================
# 4.  HYBRID CAM  =  Attention Rollout × Grad-CAM
# ===========================================================================

def compute_hybrid_cam(
    rollout: np.ndarray,
    gradcam: np.ndarray,
) -> np.ndarray:
    """
    Elementwise multiplication of Attention Rollout and Grad-CAM.

    - Attention Rollout is anatomically global but spatially diffuse.
    - Grad-CAM is spatially precise but picks up any discriminative region.
    - Their product is both anatomically grounded AND spatially precise.

    Both inputs must have the same shape (e.g. [128, 128, 128]).
    Returns normalised to [0, 1].
    """
    assert rollout.shape == gradcam.shape, (
        f"Shape mismatch: rollout {rollout.shape} ≠ gradcam {gradcam.shape}"
    )
    hybrid = rollout * gradcam
    vmin, vmax = hybrid.min(), hybrid.max()
    if vmax > vmin:
        hybrid = (hybrid - vmin) / (vmax - vmin)
    return hybrid


# ===========================================================================
# 5.  VISUALISATION — 12-panel figure
# ===========================================================================

_PANEL_CMAPS = {
    "mri":     "gray",
    "rollout": "hot",
    "gradcam": "jet",
    "hybrid":  "plasma",
}

_SLICE_KWARGS = dict(interpolation="nearest", aspect="equal")


def _mid(vol, axis):
    """Return the central slice along `axis`."""
    idx = vol.shape[axis] // 2
    return np.take(vol, idx, axis=axis)


def save_v2_visualization(
    img_tensor: torch.Tensor,               # [1,1,D,H,W]
    rollout: np.ndarray,                    # [D,H,W]
    gradcam: np.ndarray,                    # [D,H,W]
    hybrid: np.ndarray,                     # [D,H,W]
    subject_id: str,
    out_dir: str,
    pred_label: str,
    true_label: str,
    confidence: float,
):
    """
    Save a 4×3 panel figure:
      Row 0: Original MRI          (axial, coronal, sagittal)
      Row 1: Attention Rollout     (axial, coronal, sagittal)
      Row 2: Grad-CAM (raw)        (axial, coronal, sagittal)
      Row 3: Hybrid CAM            (axial, coronal, sagittal)
    """
    os.makedirs(out_dir, exist_ok=True)

    vol = img_tensor[0, 0].cpu().numpy()   # [D, H, W]

    # --- Build central slices for each map ---
    rows = {
        "Original MRI":       (vol,     _PANEL_CMAPS["mri"],     None),
        "Attention Rollout":  (rollout, _PANEL_CMAPS["rollout"], (0, 1)),
        "Grad-CAM":           (gradcam, _PANEL_CMAPS["gradcam"], (0, 1)),
        "Hybrid CAM":         (hybrid,  _PANEL_CMAPS["hybrid"],  (0, 1)),
    }
    axes_labels = ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"]

    fig = plt.figure(figsize=(15, 18))
    fig.patch.set_facecolor("#0d0d0d")
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.05)

    title = (
        f"Subject: {subject_id}  |  Prediction: {pred_label}  "
        f"({confidence:.1%})  |  True: {true_label}"
    )
    fig.suptitle(title, color="white", fontsize=13, y=0.98)

    for row_idx, (row_name, (data, cmap, vrange)) in enumerate(rows.items()):
        slices = [_mid(data, 0), _mid(data, 1), _mid(data, 2)]
        vmin = vrange[0] if vrange else data.min()
        vmax = vrange[1] if vrange else data.max()

        for col_idx, (sl, ax_lbl) in enumerate(zip(slices, axes_labels)):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax, **_SLICE_KWARGS)
            ax.set_axis_off()

            if col_idx == 0:
                ax.set_ylabel(row_name, color="white", fontsize=9, labelpad=4)
                ax.yaxis.set_label_position("left")
                ax.yaxis.label.set_visible(True)

            if row_idx == 0:
                ax.set_title(ax_lbl, color="#aaaaaa", fontsize=8, pad=3)

    out_path = os.path.join(out_dir, f"{subject_id}_v2_explainability.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [✓] Saved: {out_path}")
    return out_path


# ===========================================================================
# 6.  MAIN
# ===========================================================================

def load_model(checkpoint_path: str) -> HCCTModel:
    model = HCCTModel(CONFIG).to(DEVICE)
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] Missing keys  ({len(missing)}): {missing[:4]} …")
    if unexpected:
        print(f"  [warn] Unexpected keys ({len(unexpected)}): {unexpected[:4]} …")
    model.eval()
    return model


def run(checkpoint: str, csv_path: str, out_dir: str, num_subjects: int):
    model   = load_model(checkpoint)
    gradcam = GradCAM3D_v2(model)

    df = pd.read_csv(csv_path)

    # Sample subjects deterministically (for reproducibility)
    rng     = random.Random(42)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    indices = indices[:num_subjects]

    label_map = {0: "CN", 1: "AD"}

    for idx in indices:
        row       = df.iloc[idx]
        img_path  = row["mri_path"]
        true_str  = row["diagnosis"]
        subject_id = os.path.splitext(os.path.basename(img_path))[0]

        print(f"\n[{idx+1}/{num_subjects}] Processing {subject_id} (true: {true_str}) …")

        try:
            tensor = preprocess_volume(img_path).to(DEVICE)
        except Exception as exc:
            print(f"  [!] Preprocessing failed: {exc}")
            continue

        # --- Prediction ---
        with torch.no_grad():
            logits = model(tensor)
        probs      = torch.softmax(logits, dim=1)[0]
        pred_idx   = probs.argmax().item()
        pred_label = label_map[pred_idx]
        confidence = probs[pred_idx].item()

        # --- Attention Rollout ---
        try:
            rollout = compute_attention_rollout(model, tensor)
        except Exception as exc:
            print(f"  [!] Rollout failed: {exc}")
            continue

        # --- Grad-CAM ---
        try:
            cam = gradcam(tensor.detach().clone(), class_idx=pred_idx)
        except Exception as exc:
            print(f"  [!] Grad-CAM failed: {exc}")
            continue

        # --- Hybrid CAM ---
        hybrid = compute_hybrid_cam(rollout, cam)

        # --- Save ---
        save_v2_visualization(
            img_tensor=tensor,
            rollout=rollout,
            gradcam=cam,
            hybrid=hybrid,
            subject_id=subject_id,
            out_dir=out_dir,
            pred_label=pred_label,
            true_label=true_str,
            confidence=confidence,
        )

    gradcam.remove_hooks()
    print("\nAll done.")


# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="HCCT Attention Rollout + Hybrid CAM")
    p.add_argument("--checkpoint",   default="models/best_model.pth")
    p.add_argument("--csv",          default=CONFIG["csv_path"])
    p.add_argument("--out_dir",      default="explainability_v2_results")
    p.add_argument("--num_subjects", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        checkpoint   = args.checkpoint,
        csv_path     = args.csv,
        out_dir      = args.out_dir,
        num_subjects = args.num_subjects,
    )
