"""
anatomical_metrics.py
=====================
Quantitative metrics for evaluating anatomical grounding of explanations.

Features:
- Average Drop/Increase in Confidence (ADC/AIC)
- Anatomical region overlap analysis
- Atlas-based evaluation
- Clinical relevance scoring
"""

import numpy as np
import pandas as pd
from scipy.ndimage import zoom, binary_dilation
from scipy.spatial.distance import dice
import nibabel as nib
import os
import torch
import torch.nn as nn


def average_drop_confidence(model, original_input, heatmap, num_samples=10, radius=3):
    """
    Compute Average Drop in Confidence (ADC) by occluding high-attention regions.

    Parameters
    ----------
    model : nn.Module
        Trained model
    original_input : torch.Tensor
        Original input volume [1, 1, D, H, W]
    heatmap : np.ndarray
        Attention heatmap [D, H, W]
    num_samples : int
        Number of occlusion samples
    radius : int
        Occlusion patch radius

    Returns
    -------
    dict
        ADC metrics
    """
    model.eval()
    device = original_input.device

    # Get original prediction
    with torch.no_grad():
        original_output = model(original_input)
        original_probs = torch.softmax(original_output, dim=1)[0]
        original_confidence = original_probs.max().item()
        original_pred = original_probs.argmax().item()

    # Find high-attention regions
    threshold = np.percentile(heatmap, 80)  # Top 20% attention
    high_attention_mask = heatmap > threshold

    # Get coordinates of high-attention voxels
    coords = np.argwhere(high_attention_mask)
    if len(coords) == 0:
        return {"adc": 0.0, "samples_used": 0}

    # Sample occlusion locations
    num_coords = min(num_samples, len(coords))
    sampled_coords = coords[np.random.choice(len(coords), num_coords, replace=False)]

    confidence_drops = []

    for coord in sampled_coords:
        # Create occlusion mask
        occluded_input = original_input.clone()
        z, y, x = coord

        # Define occlusion region
        z_min = max(0, z - radius)
        z_max = min(occluded_input.shape[2], z + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(occluded_input.shape[3], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(occluded_input.shape[4], x + radius + 1)

        # Apply occlusion
        occluded_input[0, 0, z_min:z_max, y_min:y_max, x_min:x_max] = 0

        # Get prediction on occluded input
        with torch.no_grad():
            occluded_output = model(occluded_input)
            occluded_probs = torch.softmax(occluded_output, dim=1)[0]
            occluded_confidence = occluded_probs[original_pred].item()

        # Compute confidence drop
        drop = original_confidence - occluded_confidence
        confidence_drops.append(drop)

    adc = np.mean(confidence_drops) if confidence_drops else 0.0

    return {
        "adc": adc,
        "mean_drop": np.mean(confidence_drops),
        "std_drop": np.std(confidence_drops),
        "samples_used": len(confidence_drops)
    }


def anatomical_region_overlap(heatmap, atlas_labels, region_mapping=None):
    """
    Compute overlap between heatmap and anatomical atlas regions.

    Parameters
    ----------
    heatmap : np.ndarray
        Attention heatmap [D, H, W]
    atlas_labels : np.ndarray
        Atlas with region labels [D, H, W]
    region_mapping : dict, optional
        Mapping from atlas labels to region names

    Returns
    -------
    dict
        Overlap metrics per region
    """
    if heatmap.shape != atlas_labels.shape:
        # Resize heatmap to match atlas
        heatmap = zoom(heatmap, np.array(atlas_labels.shape) / np.array(heatmap.shape), order=1)

    # Threshold heatmap
    threshold = np.percentile(heatmap, 75)
    binary_heatmap = heatmap > threshold

    unique_labels = np.unique(atlas_labels)
    overlap_metrics = {}

    for label in unique_labels:
        if label == 0:  # Background
            continue

        region_name = region_mapping.get(label, f"region_{label}") if region_mapping else f"region_{label}"

        # Binary mask for this atlas region
        atlas_mask = atlas_labels == label

        # Compute overlap
        intersection = np.sum(binary_heatmap & atlas_mask)
        heatmap_sum = np.sum(binary_heatmap)
        atlas_sum = np.sum(atlas_mask)

        dice_score = 2 * intersection / (heatmap_sum + atlas_sum) if (heatmap_sum + atlas_sum) > 0 else 0
        jaccard = intersection / (heatmap_sum + atlas_sum - intersection) if (heatmap_sum + atlas_sum - intersection) > 0 else 0
        precision = intersection / heatmap_sum if heatmap_sum > 0 else 0
        recall = intersection / atlas_sum if atlas_sum > 0 else 0

        overlap_metrics[region_name] = {
            "dice": dice_score,
            "jaccard": jaccard,
            "precision": precision,
            "recall": recall,
            "atlas_voxels": atlas_sum,
            "heatmap_voxels": intersection
        }

    return overlap_metrics


def clinical_relevance_score(heatmap, clinical_regions):
    """
    Score heatmap based on focus on clinically relevant regions.

    Parameters
    ----------
    heatmap : np.ndarray
        Attention heatmap [D, H, W]
    clinical_regions : dict
        Dictionary mapping region names to binary masks

    Returns
    -------
    dict
        Clinical relevance metrics
    """
    # Normalize heatmap
    if heatmap.max() > heatmap.min():
        norm_heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        norm_heatmap = heatmap

    total_attention = np.sum(norm_heatmap)
    clinical_attention = 0
    region_scores = {}

    for region_name, region_mask in clinical_regions.items():
        if region_mask.shape != heatmap.shape:
            region_mask = zoom(region_mask.astype(float),
                             np.array(heatmap.shape) / np.array(region_mask.shape),
                             order=1) > 0.5

        region_attention = np.sum(norm_heatmap * region_mask)
        clinical_attention += region_attention

        region_scores[region_name] = {
            "attention_mass": region_attention,
            "relative_attention": region_attention / total_attention if total_attention > 0 else 0,
            "region_size": np.sum(region_mask)
        }

    clinical_score = clinical_attention / total_attention if total_attention > 0 else 0

    return {
        "clinical_score": clinical_score,
        "total_attention": total_attention,
        "clinical_attention": clinical_attention,
        "region_scores": region_scores
    }


def shortcut_detection_score(heatmap, non_clinical_regions):
    """
    Detect and score reliance on non-clinical shortcut regions.

    Parameters
    ----------
    heatmap : np.ndarray
        Attention heatmap [D, H, W]
    non_clinical_regions : dict
        Dictionary mapping shortcut region names to binary masks

    Returns
    -------
    dict
        Shortcut detection metrics
    """
    # Normalize heatmap
    if heatmap.max() > heatmap.min():
        norm_heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        norm_heatmap = heatmap

    total_attention = np.sum(norm_heatmap)
    shortcut_attention = 0
    shortcut_scores = {}

    for region_name, region_mask in non_clinical_regions.items():
        if region_mask.shape != heatmap.shape:
            region_mask = zoom(region_mask.astype(float),
                             np.array(heatmap.shape) / np.array(region_mask.shape),
                             order=1) > 0.5

        region_attention = np.sum(norm_heatmap * region_mask)
        shortcut_attention += region_attention

        shortcut_scores[region_name] = {
            "attention_mass": region_attention,
            "relative_attention": region_attention / total_attention if total_attention > 0 else 0,
            "region_size": np.sum(region_mask)
        }

    shortcut_score = shortcut_attention / total_attention if total_attention > 0 else 0

    return {
        "shortcut_score": shortcut_score,
        "total_attention": total_attention,
        "shortcut_attention": shortcut_attention,
        "region_scores": shortcut_scores
    }


def create_ad_hoc_atlas(volume_shape=(128, 128, 128)):
    """
    Create a simple anatomical atlas for evaluation when real atlas unavailable.

    Parameters
    ----------
    volume_shape : tuple
        Shape of the volume

    Returns
    -------
    dict
        Dictionary with anatomical region masks
    """
    D, H, W = volume_shape

    # Define anatomical regions (approximate locations)
    regions = {}

    # Medial temporal lobe (hippocampus, entorhinal cortex)
    mt_mask = np.zeros(volume_shape, dtype=bool)
    mt_mask[int(0.4*D):int(0.7*D), int(0.3*H):int(0.7*H), int(0.3*W):int(0.7*W)] = True
    regions['medial_temporal'] = mt_mask

    # Ventricles (lateral ventricles)
    vent_mask = np.zeros(volume_shape, dtype=bool)
    vent_mask[int(0.3*D):int(0.7*D), int(0.45*H):int(0.55*H), int(0.4*W):int(0.6*W)] = True
    regions['ventricles'] = vent_mask

    # Cerebellum (inferior region)
    cereb_mask = np.zeros(volume_shape, dtype=bool)
    cereb_mask[:int(0.25*D), :, :] = True
    regions['cerebellum'] = cereb_mask

    # Brainstem
    bs_mask = np.zeros(volume_shape, dtype=bool)
    bs_mask[:int(0.2*D), int(0.4*H):int(0.6*H), int(0.4*W):int(0.6*W)] = True
    regions['brainstem'] = bs_mask

    # FOV edges (peripheral regions)
    edge_mask = np.zeros(volume_shape, dtype=bool)
    edge_width = 8
    edge_mask[:edge_width, :, :] = True
    edge_mask[-edge_width:, :, :] = True
    edge_mask[:, :edge_width, :] = True
    edge_mask[:, -edge_width:, :] = True
    edge_mask[:, :, :edge_width] = True
    edge_mask[:, :, -edge_width:] = True
    regions['fov_edges'] = edge_mask

    return regions


def comprehensive_anatomical_evaluation(
    model,
    input_tensor,
    heatmap,
    atlas_regions=None,
    subject_id="unknown"
):
    """
    Run comprehensive anatomical evaluation of explanation quality.

    Parameters
    ----------
    model : nn.Module
        Trained model
    input_tensor : torch.Tensor
        Input volume
    heatmap : np.ndarray
        Attention heatmap
    atlas_regions : dict, optional
        Anatomical atlas regions
    subject_id : str
        Subject identifier

    Returns
    -------
    dict
        Complete evaluation results
    """
    if atlas_regions is None:
        atlas_regions = create_ad_hoc_atlas(heatmap.shape)

    results = {
        "subject_id": subject_id,
        "heatmap_stats": {
            "max": float(heatmap.max()),
            "mean": float(heatmap.mean()),
            "std": float(heatmap.std()),
            "sparsity": float(np.mean(heatmap == 0))
        }
    }

    # ADC evaluation
    adc_results = average_drop_confidence(model, input_tensor, heatmap)
    results["adc_metrics"] = adc_results

    # Clinical relevance
    clinical_regions = {k: v for k, v in atlas_regions.items()
                       if k in ['medial_temporal', 'ventricles']}
    clinical_results = clinical_relevance_score(heatmap, clinical_regions)
    results["clinical_relevance"] = clinical_results

    # Shortcut detection
    shortcut_regions = {k: v for k, v in atlas_regions.items()
                       if k in ['cerebellum', 'brainstem', 'fov_edges']}
    shortcut_results = shortcut_detection_score(heatmap, shortcut_regions)
    results["shortcut_detection"] = shortcut_results

    # Anatomical region overlap
    overlap_results = anatomical_region_overlap(heatmap, np.zeros_like(heatmap))  # Placeholder
    results["region_overlap"] = overlap_results

    # Overall scores
    results["overall_scores"] = {
        "anatomical_grounding_score": clinical_results["clinical_score"] - shortcut_results["shortcut_score"],
        "adc_score": adc_results["adc"],
        "confidence": clinical_results["clinical_score"]
    }

    return results
