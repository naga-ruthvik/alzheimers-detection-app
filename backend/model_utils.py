import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom, binary_fill_holes, binary_dilation
from einops import rearrange
import cv2

# --- Model Definitions ---
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

        weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = F.relu(cam)

        cam = cam.cpu().detach().numpy()
        cam = zoom(
            cam,
            (
                input_tensor.shape[2] / cam.shape[0],
                input_tensor.shape[3] / cam.shape[1],
                input_tensor.shape[4] / cam.shape[2],
            ),
            order=1,
        )

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output


# --- Preprocessing Helpers ---
def robust_normalize(img):
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip(img, p1, p99)
    mean, std = img.mean(), img.std()
    if std > 0:
        img = (img - mean) / std
    return img


def skull_strip(img, margin=5):
    threshold = np.percentile(img[img > 0], 30) if img.max() > 0 else img.mean()
    mask = img > threshold
    mask = binary_fill_holes(mask)
    mask = binary_dilation(mask, iterations=2)
    img = img * mask.astype(np.float32)

    coords = np.argwhere(mask)
    if len(coords) == 0:
        return img
    min_c = np.maximum(coords.min(axis=0) - margin, 0)
    max_c = np.minimum(coords.max(axis=0) + margin + 1, np.array(img.shape))
    return img[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]


def zero_border(img, border=8):
    img[:border, :, :] = 0
    img[-border:, :, :] = 0
    img[:, :border, :] = 0
    img[:, -border:, :] = 0
    img[:, :, :border] = 0
    img[:, :, -border:] = 0
    return img


def preprocess_nifti_file(file_path, size=128):
    img_nii = nib.load(file_path)
    img = img_nii.get_fdata()
    if len(img.shape) > 3:
        img = np.squeeze(img)
    img = img.astype("float32")
    img = skull_strip(img)
    img = robust_normalize(img)
    factors = (size / img.shape[0], size / img.shape[1], size / img.shape[2])
    img = zoom(img, factors, order=1)
    img = zero_border(img, border=8)
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0), img
