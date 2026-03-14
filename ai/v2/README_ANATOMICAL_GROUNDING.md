# Anatomical Grounding Implementation for HCCT Alzheimer's Detection

This repository contains a comprehensive implementation to fix shortcut learning in 3D Hybrid CNN-Transformer models for Alzheimer's disease classification, ensuring anatomically valid and clinically relevant explanations.

## Problem Statement

The original HCCT model showed high validation accuracy but suffered from **shortcut learning**, where Grad-CAM revealed attention focused on:
- Cerebellum and brainstem (non-clinical regions)
- FOV edges and scan artifacts
- Rather than clinically relevant areas like hippocampus, ventricles, and temporal cortex

## Solution Overview

### 1. Enhanced Preprocessing (`enhanced_preprocessing.py`)
- **Supratentorial brain isolation**: Advanced skull stripping that keeps only brain tissue above tentorium
- **3D Random Affine transformations**: Introduces rotation, translation, and scaling to break reliance on fixed scan artifacts
- **Anatomical region masking**: Adversarial training by randomly masking non-clinical regions

### 2. Attention Rollout for Hybrid Architecture (`attention_rollout_hybrid.py`)
- **Transformer attention flow**: Captures self-attention patterns across all layers
- **Spatial projection**: Projects 512 transformer tokens back to 128³ spatial volume
- **Hybrid CNN-Transformer attention**: Combines rollout with Grad-CAM for comprehensive explanations

### 3. Thresholded Grad-CAM (`thresholded_gradcam.py`)
- **Adaptive thresholding**: Removes low-confidence noise ("blue halo" effect)
- **Morphological cleaning**: Eliminates spurious activations
- **Confidence scoring**: Quantifies explanation quality

### 4. Anatomical Metrics (`anatomical_metrics.py`)
- **Average Drop in Confidence (ADC)**: Measures importance of attention regions
- **Anatomical region overlap**: Dice scores against clinical regions
- **Clinical relevance scoring**: Focus on hippocampus, ventricles, temporal cortex
- **Shortcut detection**: Identifies problematic attention to cerebellum/edges

### 5. NIfTI Export (`nii_saver.py`)
- **3D heatmap saving**: Export heatmaps as NIfTI for ITK-SNAP/3D Slicer
- **Overlay creation**: Combine original MRI with attention maps
- **Batch processing**: Save multiple heatmaps per subject

### 6. Enhanced Model Saving (`new_best_model_saver.py`)
- **Multi-criteria best models**: Track best by accuracy, F1, anatomical grounding
- **Comprehensive metadata**: Include evaluation metrics in checkpoints
- **Training history**: JSON/CSV logs of model performance over time

## Installation

```bash
pip install torch nibabel numpy scipy scikit-learn pandas matplotlib tqdm einops
```

## Usage

### Training with Anatomical Grounding

```python
from train_oasis_enhanced import train_enhanced_model

# Run enhanced training with anatomical evaluation
train_enhanced_model()
```

### Explainability Analysis

```python
from attention_rollout_hybrid import compute_attention_rollout_hybrid
from thresholded_gradcam import ThresholdedGradCAM3D
from nii_saver import save_heatmap_as_nifti

# Load trained model
model = HCCTModel(CONFIG)
checkpoint = torch.load("models_enhanced/best_anatomical_grounding.pth")
model.load_state_dict(checkpoint["model_state_dict"])

# Preprocess input
input_tensor = preprocess_enhanced(nifti_path)

# Compute explanations
rollout = compute_attention_rollout_hybrid(model, input_tensor)
gradcam = ThresholdedGradCAM3D(model, model.cnn[-2].conv)
thresholded_cam, _ = gradcam.generate(input_tensor)

# Save as NIfTI
save_heatmap_as_nifti(rollout, nifti_path, "attention_rollout.nii.gz")
save_heatmap_as_nifti(thresholded_cam, nifti_path, "thresholded_gradcam.nii.gz")
```

### Anatomical Evaluation

```python
from anatomical_metrics import comprehensive_anatomical_evaluation

# Evaluate explanation quality
atlas = create_ad_hoc_atlas()
results = comprehensive_anatomical_evaluation(model, input_tensor, rollout, atlas)

print(f"Anatomical Grounding Score: {results['overall_scores']['anatomical_grounding_score']:.3f}")
print(f"Clinical Relevance: {results['clinical_relevance']['clinical_score']:.3f}")
```

## Key Improvements

### Preprocessing Enhancements
1. **Supratentorial isolation**: Masks cerebellum/brainstem during training
2. **3D affine augmentation**: Prevents edge artifact reliance
3. **Adversarial region masking**: Forces attention to clinical areas

### Explainability Upgrades
1. **Attention Rollout**: Captures transformer self-attention flow
2. **Spatial projection**: Maps tokens to 3D anatomical space
3. **Thresholded Grad-CAM**: Eliminates noise and spurious activations
4. **Hybrid attention**: Combines CNN gradients with transformer attention

### Training Strategy
1. **Anatomical evaluation during training**: Monitor grounding every N epochs
2. **Multi-criteria model saving**: Best models by accuracy, F1, and anatomical scores
3. **Adversarial erasing**: Random masking of shortcut regions

### Quantitative Metrics
1. **ADC (Average Drop in Confidence)**: Importance of attention regions
2. **Region overlap analysis**: Dice scores with anatomical atlas
3. **Clinical relevance scoring**: Focus on AD biomarkers
4. **Shortcut detection**: Identifies problematic attention patterns

## Output Files

### Training Outputs
- `models_enhanced/`: Enhanced model checkpoints
- `models_log.json`: Training history and best models
- `training_summary.csv`: Performance metrics over time
- `best_*.pth`: Symlinks to best models by different criteria

### Explainability Outputs
- `attention_rollout.nii.gz`: 3D attention flow visualization
- `thresholded_gradcam.nii.gz`: Clean Grad-CAM heatmaps
- `hybrid_attention.nii.gz`: Combined explanations
- `*_overlay.nii.gz`: Original MRI with heatmap overlay

## Expected Results

After implementing these changes:

1. **Reduced shortcut learning**: Less attention to cerebellum/FOV edges
2. **Improved anatomical grounding**: Focus on hippocampus, ventricles, temporal cortex
3. **Better clinical relevance**: Explanations align with AD biomarkers
4. **Enhanced interpretability**: Clearer, more confident heatmaps
5. **Quantitative validation**: Metrics to prove anatomical validity

## Clinical Validation

The improved model should show:
- **Higher ADC scores**: Important regions significantly impact predictions
- **Better region overlap**: Strong alignment with clinical AD biomarkers
- **Lower shortcut scores**: Reduced attention to non-clinical areas
- **Improved diagnostic confidence**: More reliable explanations for clinicians

## File Structure

```
├── train_oasis_enhanced.py          # Enhanced training script
├── enhanced_preprocessing.py        # Advanced preprocessing pipeline
├── attention_rollout_hybrid.py      # Hybrid attention explanations
├── thresholded_gradcam.py           # Clean Grad-CAM implementation
├── anatomical_metrics.py            # Quantitative evaluation metrics
├── nii_saver.py                     # NIfTI export for 3D visualization
├── new_best_model_saver.py          # Advanced model checkpointing
├── explainability_v2.py             # Original explainability (kept for comparison)
├── test_and_visualize.py            # Original Grad-CAM (kept for comparison)
└── README.md                        # This documentation
```

## Citation

If you use this implementation, please cite the anatomical grounding approach for medical AI interpretability.</content>
<parameter name="filePath">c:\Users\konde\main-projects\Alzheimers Detection\README_ANATOMICAL_GROUNDING.md