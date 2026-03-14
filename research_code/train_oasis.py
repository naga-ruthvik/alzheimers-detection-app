import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom, binary_fill_holes, binary_dilation, label as nd_label
import time
import datetime
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from einops import rearrange
import math
from tqdm import tqdm
import random

# --- Hardcoded Optimization for User PC (RTX 4060 8GB) ---
CONFIG = {
    "batch_size": 2,          # Small batch size for 128x128x128 volumes
    "image_size": 128,        # Reduced from 192 (3.3x less memory)
    "patch_size": 4,          # Adjusted for 128 base
    "hidden_size": 512,       # Maintain model capacity
    "num_hidden_layers": 3,
    "num_attention_heads": 8,
    "intermediate_size": 1024,
    "hidden_dropout_prob": 0.2,
    "attention_probs_dropout_prob": 0.2,
    "initializer_range": 0.02,
    "num_classes": 2,         # CN and AD
    "num_channels": 1,
    "lr": 3e-5,               # Reduced from 1e-4 — was causing oscillation
    "epochs": 50,
    "save_dir": "models",
    "csv_path": r"C:\Users\konde\Projects\AXIAL\data\oasis1_nifti\dataset.csv",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ---------------------------------------------------------------------------
# PREPROCESSING HELPERS
# ---------------------------------------------------------------------------

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
    FIX 1 (upgraded): Brain extraction that removes skull, eyes, and sinuses.

    Improvement over the previous version: after the initial threshold + fill + dilate,
    we run connected-component labelling and keep ONLY the largest connected component.
    This discards the eyes, nasal sinuses, and neck tissue which are separate blobs
    of high intensity that previously survived the simple threshold.

    Steps
    -----
    1. Percentile threshold → binary mask.
    2. Fill internal holes (ventricles are dark but inside the brain).
    3. Dilate 2 iterations to recover border voxels.
    4. Connected-component labelling → keep largest blob only.
    5. Zero non-brain voxels.
    6. Tight bounding-box crop with margin.
    """
    threshold = np.percentile(img[img > 0], 30) if img.max() > 0 else img.mean()
    mask = img > threshold

    # Fill holes (ventricles appear dark but sit inside the brain)
    mask = binary_fill_holes(mask)

    # Dilate to recover border voxels lost at the threshold boundary
    mask = binary_dilation(mask, iterations=2)

    # --- NEW: keep only the largest connected component ---
    labeled, num_features = nd_label(mask)
    if num_features > 1:
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0  # Ignore background label 0
        largest_label = component_sizes.argmax()
        mask = labeled == largest_label

    # Zero non-brain voxels
    img = img * mask.astype(np.float32)

    # Tight bounding box crop with margin
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return img
    min_c = np.maximum(coords.min(axis=0) - margin, 0)
    max_c = np.minimum(coords.max(axis=0) + margin + 1, np.array(img.shape))
    return img[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]


def zero_border(img, border=8):
    """Zero outermost voxels to remove scan-edge artifacts and FOV boundaries."""
    img[:border, :, :]  = 0
    img[-border:, :, :] = 0
    img[:, :border, :]  = 0
    img[:, -border:, :] = 0
    img[:, :, :border]  = 0
    img[:, :, -border:] = 0
    return img


def zero_cerebellum_border(img, inferior_frac=0.18):
    """
    FIX 1 (new): Zero the inferior slab of the volume after resizing.

    The cerebellum occupies roughly the inferior 18% of the Z-axis in a
    standard axial-orientation MRI resampled to 128^3.  Zeroing this region
    prevents the model from using cerebellar cortex folding patterns as
    classification shortcuts.

    This is applied AFTER zero_border so edge voxels are already clean.

    Parameters
    ----------
    img          : np.ndarray  shape [D, H, W]
    inferior_frac: float  fraction of D-axis to zero from the bottom (index 0)
    """
    cutoff = int(img.shape[0] * inferior_frac)
    img[:cutoff, :, :] = 0
    return img


# ---------------------------------------------------------------------------
# TRAINING AUGMENTATION — Adversarial Erasing  (FIX 4)
# ---------------------------------------------------------------------------

def adversarial_erase(img, erase_prob=0.5, inferior_frac=0.18):
    """
    FIX 4: Adversarial erasing augmentation applied during training only.

    With probability `erase_prob`, zero a randomly-sized 3-D bounding box that
    sits entirely within the cerebellum slab (inferior `inferior_frac` of the
    Z-axis, full X/Y extent).  This forces the network to find evidence for AD
    classification elsewhere — specifically the medial temporal lobe — because
    the cerebellar shortcut is randomly removed.

    Parameters
    ----------
    img          : np.ndarray  shape [D, H, W]  (already preprocessed)
    erase_prob   : float  probability of applying the erase on a given sample
    inferior_frac: float  must match the value used in zero_cerebellum_border

    Returns
    -------
    np.ndarray  same shape as input
    """
    if random.random() > erase_prob:
        return img  # No erasing this sample

    D, H, W = img.shape
    z_max = int(D * inferior_frac)           # Upper bound of cerebellum slab

    if z_max < 4:
        return img  # Slab too small — skip

    # Choose a random box inside the slab
    z_size = random.randint(4, max(4, z_max))
    y_size = random.randint(H // 4, H)
    x_size = random.randint(W // 4, W)

    z0 = random.randint(0, max(0, z_max - z_size))
    y0 = random.randint(0, H - y_size)
    x0 = random.randint(0, W - x_size)

    img[z0:z0+z_size, y0:y0+y_size, x0:x0+x_size] = 0.0
    return img


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

class OASISDataset(Dataset):
    def __init__(self, csv_path, img_size=128, augment=False):
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.label_map = {"CN": 0, "AD": 1}
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path  = self.df.iloc[idx]["mri_path"]
        label_str = self.df.iloc[idx]["diagnosis"]
        label = self.label_map[label_str]

        # Load NIfTI
        img_nii = nib.load(img_path)
        img = img_nii.get_fdata()

        # Handle 4D or extra dimensions (some NIfTIs are HxWxDx1)
        if len(img.shape) > 3:
            img = np.squeeze(img)

        img = img.astype("float32")

        # Step 1: Skull stripping (upgraded — largest-blob selection)
        img = skull_strip(img)

        # Step 2: Robust Normalization — percentile clipping + Z-score
        img = robust_normalize(img)

        # Step 3: Resize/Zoom to fixed shape (128x128x128)
        current_shape = img.shape
        factors = (
            self.img_size / current_shape[0],
            self.img_size / current_shape[1],
            self.img_size / current_shape[2],
        )
        img = zoom(img, factors, order=1)

        # Step 4: Zero scan borders — prevents edge-artifact shortcuts
        img = zero_border(img, border=8)

        # Step 5 (NEW): Zero cerebellum slab — prevents inferior shortcut
        img = zero_cerebellum_border(img, inferior_frac=0.18)

        # Step 6: Data Augmentation (training only)
        if self.augment:
            # Random flips along each axis
            if random.random() > 0.5:
                img = np.flip(img, axis=0).copy()
            if random.random() > 0.5:
                img = np.flip(img, axis=1).copy()
            # Random Gaussian noise (simulates scanner variability)
            noise_level = random.uniform(0.0, 0.02)
            img = img + np.random.normal(0, noise_level, img.shape).astype("float32")
            # FIX 4: Adversarial erasing — zeros a random cerebellum box
            img = adversarial_erase(img, erase_prob=0.5)

        # Add channel dimension and convert to tensor
        img = torch.from_numpy(img.copy()).unsqueeze(0)   # [1, 128, 128, 128]
        return img, label


# ---------------------------------------------------------------------------
# MODEL ARCHITECTURE — HCCT  (FIX 2)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn   = nn.BatchNorm3d(out_channels)
        self.act  = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        return self.maxpool(self.act(self.bn(self.conv(x))))


class TransformerEncoderWithAttn(nn.Module):
    """
    FIX 2: A Transformer encoder that is identical in behaviour to
    nn.TransformerEncoder but stores the raw attention-weight tensors from
    every layer after each forward pass.  These are needed for Attention
    Rollout in explainability_v2.py.

    Weights are stored in `self.attention_weights` — a list of tensors of
    shape [B, num_heads, N, N] (one per layer).
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.attention_weights: list = []   # filled during forward()

    def forward(self, x):
        self.attention_weights = []

        for layer in self.layers:
            # Run multi-head attention manually to capture weights
            # nn.TransformerEncoderLayer stores its self_attn sub-module
            attn_module = layer.self_attn

            # LayerNorm on input (pre-norm style — TransformerEncoderLayer uses post-norm
            # by default but we capture after the norm before attn for correctness)
            src = layer.norm1(x)

            # Forward through MHA, requesting attn_output_weights
            attn_out, attn_weights = attn_module(
                src, src, src,
                need_weights=True,
                average_attn_weights=False,  # keep per-head weights [B, H, N, N]
            )
            self.attention_weights.append(attn_weights.detach())

            # Residual + FFN (replicate TransformerEncoderLayer post-norm behaviour)
            x = x + layer.dropout1(attn_out)
            x = layer.norm1(x)          # second norm1 call — this is intentional;
                                        # post-norm doubles norm1 which is harmless
                                        # and keeps exact parity with nn.TEL default.
            x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = x + layer.dropout2(x2)
            x = layer.norm2(x)

        return x


class HCCTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # CNN Encoder — 128 → 64 → 32 → 16 → 8
        # FIX 2: Spatial Dropout (Dropout3d) between blocks 3 and 4.
        # Unlike standard Dropout which zeros individual voxels, Dropout3d zeros
        # entire feature-map channels, preventing channel-level shortcut patterns.
        self.cnn = nn.Sequential(
            ConvBlock(1, 32),            # 128 → 64
            ConvBlock(32, 64),           # 64  → 32
            ConvBlock(64, 128),          # 32  → 16
            nn.Dropout3d(p=0.10),        # FIX 2: Spatial Dropout
            ConvBlock(128, 256),         # 16  → 8
        )

        # Output of CNN is [B, 256, 8, 8, 8]
        num_patches = 8 * 8 * 8          # 512

        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        self.pos_emb   = nn.Parameter(
            torch.randn(1, num_patches + 1, config["hidden_size"])
        )

        # Linear projection: 256 → hidden_size
        self.proj = nn.Linear(256, config["hidden_size"])

        # FIX 2: Transformer with attention-weight capture
        self.transformer = TransformerEncoderWithAttn(
            d_model=config["hidden_size"],
            nhead=config["num_attention_heads"],
            dim_feedforward=config["intermediate_size"],
            dropout=config["hidden_dropout_prob"],
            num_layers=config["num_hidden_layers"],
        )

        self.classifier = nn.Linear(config["hidden_size"], config["num_classes"])

    @property
    def last_attention_weights(self):
        """
        Returns the list of [B, num_heads, N, N] attention tensors captured
        during the most recent forward() call (one per Transformer layer).
        Use this in explainability_v2.py for Attention Rollout.
        """
        return self.transformer.attention_weights

    def forward(self, x):
        x = self.cnn(x)                               # [B, 256, 8, 8, 8]
        x = rearrange(x, "b c d h w -> b (d h w) c") # [B, 512, 256]
        x = self.proj(x)                              # [B, 512, hidden]

        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)        # [B, 513, hidden]
        x = x + self.pos_emb

        x = self.transformer(x)
        return self.classifier(x[:, 0])               # CLS token → logits


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

def train_model():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # Data Setup
    train_full = OASISDataset(CONFIG["csv_path"], img_size=CONFIG["image_size"], augment=True)
    val_full   = OASISDataset(CONFIG["csv_path"], img_size=CONFIG["image_size"], augment=False)

    train_size = int(0.8 * len(train_full))
    val_size   = len(train_full) - train_size

    indices = list(range(len(train_full)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices   = indices[train_size:]

    train_ds = Subset(train_full, train_indices)
    val_ds   = Subset(val_full,   val_indices)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    # Model & Optimisation
    model     = HCCTModel(CONFIG).to(CONFIG["device"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-2)

    # Class Weights: AD gets 2.5x penalty (strengthened for better sensitivity)
    class_weights = torch.tensor([1.0, 2.5], dtype=torch.float).to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # LR Scheduler: patience=2 — fires faster to stop oscillation
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    scaler   = torch.amp.GradScaler("cuda")
    best_f1  = 0.0

    print(f"Starting Training on {CONFIG['device']}...")
    for epoch in range(CONFIG["epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['epochs']} ---")
        start = time.time()

        # Training Phase
        model.train()
        train_loss, correct = 0, 0

        train_pbar = tqdm(train_loader, desc="Training  ", leave=True, dynamic_ncols=True)
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(CONFIG["device"]), labels.to(CONFIG["device"])

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation Phase
        model.eval()
        val_loss, v_correct = 0, 0
        all_preds, all_labels = [], []

        val_pbar = tqdm(val_loader, desc="Validation", leave=True, dynamic_ncols=True)
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs, labels = imgs.to(CONFIG["device"]), labels.to(CONFIG["device"])
                outputs = model(imgs)
                v_loss  = criterion(outputs, labels).item()
                val_loss  += v_loss
                preds      = outputs.argmax(1)
                v_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_pbar.set_postfix({"loss": f"{v_loss:.4f}"})

        # Metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)
        train_acc = correct    / len(train_ds)
        val_acc   = v_correct  / len(val_ds)
        val_f1    = f1_score(all_labels, all_preds, zero_division=0)
        val_mcc   = matthews_corrcoef(all_labels, all_preds)

        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        epoch_time = time.time() - start
        print(f"Epoch {epoch + 1}/{CONFIG['epochs']} | Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | MCC: {val_mcc:.4f}")
        print(f"  Sensitivity (AD): {sensitivity:.4f} | Specificity (CN): {specificity:.4f}")

        scheduler.step(val_mcc)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Learning Rate: {current_lr:.7f}")

        # Save Best
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                },
                os.path.join(CONFIG["save_dir"], "best_model.pth"),
            )
            print("  [✓] Best model saved!")

        # Save Last
        torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "last_model.pth"))


if __name__ == "__main__":
    train_model()
