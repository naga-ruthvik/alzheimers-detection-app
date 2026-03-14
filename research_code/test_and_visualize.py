import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom, binary_fill_holes, binary_dilation
import matplotlib.pyplot as plt
from einops import rearrange
import cv2
from tqdm import tqdm
import random


# --- Model Definitions (Must match train_oasis.py) ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        return self.maxpool(self.act(self.bn(self.conv(x))))


class HCCTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        num_patches = 8 * 8 * 8
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        self.pos_emb = nn.Parameter(
            torch.randn(1, num_patches + 1, config["hidden_size"])
        )
        self.proj = nn.Linear(256, config["hidden_size"])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["hidden_size"],
            nhead=config["num_attention_heads"],
            dim_feedforward=config["intermediate_size"],
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config["num_hidden_layers"]
        )
        self.classifier = nn.Linear(config["hidden_size"], config["num_classes"])

    def forward(self, x):
        x = self.cnn(x)
        x = rearrange(x, "b c d h w -> b (d h w) c")
        x = self.proj(x)
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_emb
        x = self.transformer(x)
        return self.classifier(x[:, 0])


# --- Grad-CAM Logic ---
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        # Pool gradients across depth, height, width
        weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)

        # Weighted sum of activations
        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = F.relu(cam)

        cam = cam.cpu().detach().numpy()
        # Upscale to original image size
        cam = zoom(
            cam,
            (
                input_tensor.shape[2] / cam.shape[0],
                input_tensor.shape[3] / cam.shape[1],
                input_tensor.shape[4] / cam.shape[2],
            ),
            order=1,
        )

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output


# --- Preprocessing Helpers (must match train_oasis.py) ---
def robust_normalize(img):
    """Percentile-based clipping + Z-score. Prevents background from dominating."""
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip(img, p1, p99)
    mean, std = img.mean(), img.std()
    if std > 0:
        img = (img - mean) / std
    return img


def skull_strip(img, margin=5):
    """
    Improved brain extraction that removes skull, eyes, and sinuses.
    Uses a percentile threshold + morphological hole-filling to build a
    tight brain mask. Voxels outside the mask are zeroed so the model
    cannot use non-brain tissue as shortcuts (Clever Hans effect).
    """
    # Use 30th percentile as threshold — brain tissue sits above air/background
    threshold = np.percentile(img[img > 0], 30) if img.max() > 0 else img.mean()
    mask = img > threshold

    # Fill internal holes (ventricles appear dark but are inside the brain)
    mask = binary_fill_holes(mask)

    # Dilate slightly to recover border voxels lost by the threshold
    mask = binary_dilation(mask, iterations=2)

    # Zero out non-brain voxels
    img = img * mask.astype(np.float32)

    # Crop bounding box with margin
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return img
    min_c = np.maximum(coords.min(axis=0) - margin, 0)
    max_c = np.minimum(coords.max(axis=0) + margin + 1, np.array(img.shape))
    return img[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]


def zero_border(img, border=8):
    """
    Zero the outermost `border` voxels on every face of the volume.
    Prevents the model from using scan-edge artifacts or FOV boundaries
    as classification shortcuts.
    """
    img[:border, :, :] = 0
    img[-border:, :, :] = 0
    img[:, :border, :] = 0
    img[:, -border:, :] = 0
    img[:, :, :border] = 0
    img[:, :, -border:] = 0
    return img


# --- Preprocessing ---
def preprocess_nifti(path, size=128):
    img_nii = nib.load(path)
    img = img_nii.get_fdata()
    if len(img.shape) > 3:
        img = np.squeeze(img)
    img = img.astype("float32")
    # Step 1: Skull stripping + anatomical cropping (removes eyes/sinus/skull)
    img = skull_strip(img)
    # Step 2: Robust Normalization
    img = robust_normalize(img)
    # Step 3: Resize to target volume
    factors = (size / img.shape[0], size / img.shape[1], size / img.shape[2])
    img = zoom(img, factors, order=1)
    # Step 4: Zero scan edges (prevents edge-artifact shortcuts)
    img = zero_border(img, border=8)
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 128, 128]


def save_visualization(
    subject_id, img_tensor, cam, prediction_probs, true_label, output_path
):
    img = img_tensor.squeeze().cpu().numpy()
    # Rescale to [0, 1] for display only (Z-score values go outside this range)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Select middle slices
    mid_idx = img.shape[0] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Subject: {subject_id} | True: {true_label} | Pred: {prediction_probs}\n(Red highlights model attention)",
        fontsize=16,
    )

    # 1. Original Slices (Axial, Coronal, Sagittal)
    axes[0, 0].imshow(np.rot90(img[mid_idx, :, :]), cmap="gray")
    axes[0, 0].set_title("Axial (Original)")
    axes[0, 1].imshow(np.rot90(img[:, mid_idx, :]), cmap="gray")
    axes[0, 1].set_title("Coronal (Original)")
    axes[0, 2].imshow(np.rot90(img[:, :, mid_idx]), cmap="gray")
    axes[0, 2].set_title("Sagittal (Original)")

    # 2. Grad-CAM Overlays
    for i, (slice_data, cam_slice, title) in enumerate(
        [
            (np.rot90(img[mid_idx, :, :]), np.rot90(cam[mid_idx, :, :]), "Axial CAM"),
            (np.rot90(img[:, mid_idx, :]), np.rot90(cam[:, mid_idx, :]), "Coronal CAM"),
            (
                np.rot90(img[:, :, mid_idx]),
                np.rot90(cam[:, :, mid_idx]),
                "Sagittal CAM",
            ),
        ]
    ):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_slice), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        # Overlay heatmap on grayscale image
        gray_img = np.stack([slice_data] * 3, axis=-1)
        overlay = 0.6 * gray_img + 0.4 * heatmap
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(title)

    for ax in axes.flatten():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# --- Main Logic ---
if __name__ == "__main__":
    CONFIG = {
        "hidden_size": 512,
        "num_hidden_layers": 3,
        "num_attention_heads": 8,
        "intermediate_size": 1024,
        "num_classes": 2,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_path": "models/best_model.pth",
        "csv_path": r"C:\Users\konde\Projects\AXIAL\data\oasis1_nifti\dataset.csv",
        "output_dir": "explainability_results",
    }

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Load Model
    model = HCCTModel(CONFIG).to(CONFIG["device"])
    checkpoint = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Target the 2nd-to-last CNN block (index -2 = 3rd block, 16x16x16 spatial res).
    # The last block (index -1) outputs 8x8x8 — too coarse after upsampling,
    # causing empty/dispersed heatmaps. The 3rd block retains 2x more spatial detail.
    target_layer = model.cnn[-2].conv
    gcam = GradCAM3D(model, target_layer)

    # Load sample data
    df = pd.read_csv(CONFIG["csv_path"])
    sample_subjects = ["OAS1_0001", "OAS1_0002", "OAS1_0003", "OAS1_0015"]
    samples = df[df["subject"].isin(sample_subjects)]

    label_map = {0: "CN (Healthy)", 1: "AD (Alzheimer's)"}

    print("Starting Explainability Analysis...")
    for _, row in tqdm(samples.iterrows(), total=len(samples)):
        sid = row["subject"]
        mri_path = row["mri_path"]
        true_diag = row["diagnosis"]

        # Preprocess
        input_tensor = preprocess_nifti(mri_path).to(CONFIG["device"])

        # Generate CAM
        cam, output = gcam.generate(input_tensor)
        probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
        pred_diag = label_map[probs.argmax()]

        prob_str = f"CN: {probs[0]:.2f}, AD: {probs[1]:.2f}"

        # Save Visualization
        output_file = os.path.join(CONFIG["output_dir"], f"{sid}_explainability.png")
        save_visualization(sid, input_tensor[0], cam, prob_str, true_diag, output_file)
        print(f"  [✓] Result saved for {sid} -> {output_file}")

    print(f"\nAnalysis complete! Check the '{CONFIG['output_dir']}' folder.")
