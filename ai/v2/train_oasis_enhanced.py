"""
train_oasis_enhanced.py
=======================
Enhanced training script with anatomical grounding and improved explainability.

Integrates:
- Enhanced preprocessing with 3D affine transforms
- Adversarial erasing for shortcut prevention
- Anatomical evaluation during training
- Advanced model saving with metrics
"""

import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import time
import datetime
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from einops import rearrange
import math
from tqdm import tqdm
import random

# Import our new modules
from .enhanced_preprocessing import enhanced_preprocessing_pipeline
from .new_best_model_saver import AnatomicalModelSaver
from .anatomical_metrics import comprehensive_anatomical_evaluation, create_ad_hoc_atlas
from .attention_rollout_hybrid import compute_attention_rollout_hybrid
from .thresholded_gradcam import ThresholdedGradCAM3D

# --- Configuration ---
CONFIG = {
    "batch_size": 2,
    "image_size": 128,
    "patch_size": 4,
    "hidden_size": 512,
    "num_hidden_layers": 3,
    "num_attention_heads": 8,
    "intermediate_size": 1024,
    "hidden_dropout_prob": 0.2,
    "attention_probs_dropout_prob": 0.2,
    "initializer_range": 0.02,
    "num_classes": 2,
    "num_channels": 1,
    "lr": 3e-5,
    "epochs": 50,
    "save_dir": "models_enhanced",
    "csv_path": r"C:\Users\konde\main-projects\alzheimers-pred\data\oasis1_nifti\dataset.csv",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "experiment_name": "hcct_anatomical_grounding",
    "anatomical_evaluation_frequency": 5,  # Evaluate every N epochs
}


# --- Model Architecture (same as before but with attention capture) ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        return self.maxpool(self.act(self.bn(self.conv(x))))


class TransformerEncoderWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu", batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.attention_weights = []

    def forward(self, x):
        self.attention_weights = []
        for layer in self.layers:
            attn_module = layer.self_attn
            src = layer.norm1(x)
            attn_out, attn_weights = attn_module(src, src, src, need_weights=True, average_attn_weights=False)
            self.attention_weights.append(attn_weights.detach())
            x = x + layer.dropout1(attn_out)
            x = layer.norm1(x)
            x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = x + layer.dropout2(x2)
            x = layer.norm2(x)
        return x


class HCCTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(1, 32), ConvBlock(32, 64), ConvBlock(64, 128),
            nn.Dropout3d(p=0.10), ConvBlock(128, 256)
        )
        num_patches = 8 * 8 * 8
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches + 1, config["hidden_size"]))
        self.proj = nn.Linear(256, config["hidden_size"])
        self.transformer = TransformerEncoderWithAttn(
            d_model=config["hidden_size"], nhead=config["num_attention_heads"],
            dim_feedforward=config["intermediate_size"], dropout=config["hidden_dropout_prob"],
            num_layers=config["num_hidden_layers"]
        )
        self.classifier = nn.Linear(config["hidden_size"], config["num_classes"])

    @property
    def last_attention_weights(self):
        return self.transformer.attention_weights

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


# --- Enhanced Dataset ---
class EnhancedOASISDataset(Dataset):
    def __init__(self, csv_path, img_size=128, augment=False):
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.label_map = {"CN": 0, "AD": 1}
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["mri_path"]
        # Correct the path to match the new data location
        img_path = img_path.replace(
            r"C:\Users\konde\Projects\AXIAL\data",
            r"C:\Users\konde\main-projects\alzheimers-pred\data"
        )
        label_str = self.df.iloc[idx]["diagnosis"]
        label = self.label_map[label_str]

        # Use enhanced preprocessing
        img = enhanced_preprocessing_pipeline(img_path, self.img_size, self.augment)
        return img, label


# --- Training Function ---
def train_enhanced_model():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # Initialize anatomical model saver
    model_saver = AnatomicalModelSaver(CONFIG["save_dir"], CONFIG["experiment_name"])

    # Data Setup
    train_full = EnhancedOASISDataset(CONFIG["csv_path"], img_size=CONFIG["image_size"], augment=True)
    val_full = EnhancedOASISDataset(CONFIG["csv_path"], img_size=CONFIG["image_size"], augment=False)

    train_size = int(0.8 * len(train_full))
    val_size = len(train_full) - train_size

    indices = list(range(len(train_full)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_ds = Subset(train_full, train_indices)
    val_ds = Subset(val_full, val_indices)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    # Model & Optimizer
    model = HCCTModel(CONFIG).to(CONFIG["device"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-2)

    class_weights = torch.tensor([1.0, 10.0], dtype=torch.float).to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda")

    best_f1 = 0.0

    print(f"Starting Enhanced Training on {CONFIG['device']}...")
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
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
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
                v_loss = criterion(outputs, labels).item()
                val_loss += v_loss
                preds = outputs.argmax(1)
                v_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_pbar.set_postfix({"loss": f"{v_loss:.4f}"})

        # Metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = correct / len(train_ds)
        val_acc = v_correct / len(val_ds)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        val_mcc = matthews_corrcoef(all_labels, all_preds)

        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        epoch_time = time.time() - start
        print(f"Epoch {epoch + 1}/{CONFIG['epochs']} | Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | MCC: {val_mcc:.4f}")
        print(f"  Sensitivity (AD): {sensitivity:.4f} | Specificity (CN): {specificity:.4f}")

        # Prepare metrics for saving
        val_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_mcc": val_mcc,
            "sensitivity": sensitivity,
            "specificity": specificity
        }

        # Anatomical Evaluation (every N epochs)
        anatomical_results = None
        if (epoch + 1) % CONFIG["anatomical_evaluation_frequency"] == 0:
            print("  Performing anatomical evaluation...")
            try:
                anatomical_results = evaluate_anatomical_grounding(
                    model, val_loader, num_samples=min(5, len(val_ds))
                )
                print(f"    Anatomical Grounding Score: {anatomical_results['overall_scores']['anatomical_grounding_score']:.3f}")
                print(f"    Clinical Relevance: {anatomical_results['clinical_relevance']['clinical_score']:.3f}")
                print(f"    Anti-Shortcut Score: {1 - anatomical_results['shortcut_detection']['shortcut_score']:.3f}")
            except Exception as e:
                print(f"    Anatomical evaluation failed: {e}")

        # Save model with anatomical evaluation
        checkpoint_path = model_saver.save_model_with_anatomical_evaluation(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics=val_metrics,
            anatomical_metrics=anatomical_results
        )

        # Save best based on F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.pth"))
            print("  [✓] Best model saved!")

        scheduler.step(val_mcc)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Learning Rate: {current_lr:.7f}")

    # Create symlinks to best models
    model_saver.save_best_model_symlinks()

    # Export training summary
    model_saver.export_training_summary()

    print(f"\nTraining complete! Models saved in '{CONFIG['save_dir']}'")


def evaluate_anatomical_grounding(model, val_loader, num_samples=5):
    """
    Evaluate anatomical grounding on validation samples.
    """
    model.eval()

    # Get a few validation samples
    sample_inputs, sample_labels = [], []
    for i, (inputs, labels) in enumerate(val_loader):
        if i >= num_samples:
            break
        sample_inputs.append(inputs)
        sample_labels.append(labels)

    if not sample_inputs:
        return None

    # Concatenate samples
    all_inputs = torch.cat(sample_inputs, dim=0)
    all_labels = torch.cat(sample_labels, dim=0)

    # Compute attention rollout for first sample
    sample_input = all_inputs[0:1].to(CONFIG["device"])
    rollout = compute_attention_rollout_hybrid(model, sample_input)

    # Compute thresholded Grad-CAM
    gradcam = ThresholdedGradCAM3D(model, model.cnn[-1].conv)
    thresholded_cam, _ = gradcam.generate(sample_input)

    # Create ad-hoc atlas
    atlas = create_ad_hoc_atlas(rollout.shape)

    # Comprehensive evaluation
    results = comprehensive_anatomical_evaluation(
        model, sample_input, rollout, atlas, subject_id="validation_sample"
    )

    return results


if __name__ == "__main__":
    train_enhanced_model()
