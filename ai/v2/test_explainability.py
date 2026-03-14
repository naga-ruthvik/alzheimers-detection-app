"""
test_explainability.py
======================
Test script for Alzheimer's detection with explainability.

Tests the trained model on OASIS-1 and OASIS-2 samples with:
- Inference predictions
- Thresholded Grad-CAM
- Attention Rollout
- Anatomical evaluation
- NIfTI export for 3D visualization
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
import random
from einops import rearrange

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.enhanced_preprocessing import enhanced_preprocessing_pipeline
from v2.thresholded_gradcam import ThresholdedGradCAM3D, combine_thresholded_gradcam_and_rollout
from v2.attention_rollout_hybrid import compute_attention_rollout_hybrid
from v2.anatomical_metrics import comprehensive_anatomical_evaluation, create_ad_hoc_atlas
from v2.nii_saver import save_heatmap_as_nifti, create_overlay_nifti, save_2d_slices_with_overlay
from v2.new_best_model_saver import AnatomicalModelSaver

# Configuration
CONFIG = {
    "batch_size": 1,  # Single sample testing
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
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "models_enhanced/best_model.pth",  # Will be updated
    "test_output_dir": "test_results",
    "dataset_csv": "dataset.csv",  # The attached CSV with both OAS1 and OAS2
}

# Model Architecture (same as training)
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

def load_best_model(config):
    """Load the best trained model."""
    # Find the best model
    saver = AnatomicalModelSaver(config["model_path"].replace("best_model.pth", ""))
    best_model_path = saver.load_best_model(criterion="f1_score", model_class=lambda: HCCTModel(config), device=config["device"])
    model, checkpoint = best_model_path

    print(f"Loaded best model: F1={checkpoint['metrics'].get('val_f1', 'N/A')}")
    return model, checkpoint

def select_test_samples(df, num_samples=3):
    """Select test samples from OASIS-1 and OASIS-2."""
    # Select num_samples OASIS-1 samples for analysis
    oas1_samples = df[df['subject'].str.startswith('OAS1')].sample(num_samples, random_state=42)
    return oas1_samples

def test_sample(model, sample, device):
    """Test a single sample and return predictions and explanations."""
    img_path = sample['mri_path']
    true_label = 0 if sample['diagnosis'] == 'CN' else 1
    subject_id = sample['subject']

    # Load and preprocess
    try:
        img = enhanced_preprocessing_pipeline(img_path, CONFIG["image_size"], augment=False)
        img = img.unsqueeze(0).to(device)  # Add batch dimension
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

    model.eval()
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()

    # Compute explanations
    try:
        # Attention Rollout
        rollout = compute_attention_rollout_hybrid(model, img)

        # Try Grad-CAM on all ConvBlocks
        gradcam_layers = [model.cnn[0].conv, model.cnn[1].conv, model.cnn[2].conv, model.cnn[4].conv]
        gradcam_results = {}
        for idx, layer in enumerate(gradcam_layers):
            gradcam = ThresholdedGradCAM3D(model, layer)
            cam, _ = gradcam.generate(img)
            gradcam_results[f'conv{idx+1}'] = cam

        # Use the last ConvBlock for combined explanation (as before)
        thresholded_cam = gradcam_results['conv4']
        combined = combine_thresholded_gradcam_and_rollout(thresholded_cam, rollout)

        # Anatomical evaluation
        atlas = create_ad_hoc_atlas(rollout.shape)
        anatomical_results = comprehensive_anatomical_evaluation(
            model, img, rollout, atlas, subject_id=subject_id
        )

    except Exception as e:
        print(f"Error in explainability for {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        rollout = thresholded_cam = combined = anatomical_results = gradcam_results = None

    return {
        'subject_id': subject_id,
        'true_label': true_label,
        'pred_class': pred_class,
        'confidence': confidence,
        'prob_CN': probs[0].item(),
        'prob_AD': probs[1].item(),
        'rollout': rollout,
        'gradcam': thresholded_cam,
        'combined': combined,
        'anatomical': anatomical_results,
        'img_path': img_path,
        'gradcam_results': gradcam_results
    }

def save_results(results, output_dir):
    """Save test results and visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    summary_data = []

    for result in results:
        if result is None:
            continue

        subject = result['subject_id']
        subject_dir = os.path.join(output_dir, subject)
        os.makedirs(subject_dir, exist_ok=True)

        # Save summary
        summary_data.append({
            'subject': subject,
            'true_label': 'CN' if result['true_label'] == 0 else 'AD',
            'pred_label': 'CN' if result['pred_class'] == 0 else 'AD',
            'confidence': result['confidence'],
            'correct': result['true_label'] == result['pred_class']
        })


        # ...existing code...

        # Save 2D slices with prediction overlay
        if result['combined'] is not None:
            pred_label = 'AD' if result['pred_class'] == 1 else 'CN'
            pred_conf = result['confidence']
            save_2d_slices_with_overlay(
                result['img_path'],
                result['combined'],
                os.path.join(subject_dir, f'{subject}_slices.png'),
                overlay_text=f"Predicted: {pred_label} | Confidence: {pred_conf:.2f}"
            )

        # Save anatomical metrics
        if result['anatomical']:
            with open(os.path.join(subject_dir, f'{subject}_anatomical_metrics.txt'), 'w') as f:
                f.write(f"Anatomical Evaluation for {subject}\n")
                f.write("=" * 50 + "\n")
                for key, value in result['anatomical'].items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for subkey, subvalue in value.items():
                            f.write(f"  {subkey}: {subvalue}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")

    # Save summary CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'test_summary.csv'), index=False)

        # Print summary
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        print(summary_df.to_string(index=False))

        accuracy = summary_df['correct'].mean()
        print(".1f")
        print(".1f")

def main():
    print("Starting Alzheimer's Detection Testing with Explainability")
    print("="*60)

    # Load dataset
    df = pd.read_csv(CONFIG["dataset_csv"])

    # Correct paths for the actual data location
    df['mri_path'] = df['mri_path'].str.replace(
        'data/oasis1_nifti/',
        r'C:\Users\konde\main-projects\alzheimers-pred\data\oasis1_nifti\\'
    )
    df['mri_path'] = df['mri_path'].str.replace(
        'data/oasis2_nifti/',
        r'C:\Users\konde\main-projects\alzheimers-pred\data\oasis2_nifti\\'
    )

    print(f"Dataset loaded: {len(df)} samples")
    print(f"OASIS-1 samples: {df['subject'].str.startswith('OAS1').sum()}")
    print(f"OASIS-2 samples: {df['subject'].str.startswith('OAS2').sum()}")

    # Select test samples
    test_samples = select_test_samples(df, num_samples=10)
    print(f"\nSelected test samples:")
    for _, row in test_samples.iterrows():
        print(f"  {row['subject']}: {row['diagnosis']} ({row['mri_path']})")

    # Load model
    try:
        model, checkpoint = load_best_model(CONFIG)
        model.to(CONFIG["device"])
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Test samples
    results = []
    for _, sample in test_samples.iterrows():
        print(f"\nTesting {sample['subject']}...")
        result = test_sample(model, sample, CONFIG["device"])
        results.append(result)

    # Save results
    save_results(results, CONFIG["test_output_dir"])

    print(f"\nTesting complete! Results saved to '{CONFIG['test_output_dir']}'")
    print("Use ITK-SNAP or 3D Slicer to view the .nii.gz files for 3D visualization.")

if __name__ == "__main__":
    main()