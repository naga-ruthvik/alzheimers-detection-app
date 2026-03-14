import os
import sys
import uuid
import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
from flask import Blueprint, request, jsonify
from PIL import Image

# Make sure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib.util

# Try normal package import first, then fall back to loading the file directly
te = None
try:
    from ai.v2 import test_explainability as te  # type: ignore
except Exception:
    te_path = os.path.join(PROJECT_ROOT, 'ai', 'v2', 'test_explainability.py')
    if os.path.exists(te_path):
        spec = importlib.util.spec_from_file_location('ai_v2_test_explainability', te_path)
        te_mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(te_mod)  # type: ignore
            te = te_mod
        except Exception as e:
            print('Failed to load ai.v2.test_explainability from file:', e)
            te = None
    else:
        te = None
else:
    print('Imported ai.v2.test_explainability via package import')

if te is not None:
    print('ai.v2.test_explainability available (te set)')
else:
    print('ai.v2.test_explainability NOT available (te is None)')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

explain_bp = Blueprint('explain', __name__)


@explain_bp.route('/predict_explain', methods=['POST'])
def predict_explain():
    if te is None:
        return jsonify({'status': 'error', 'message': 'ai.v2 module not available'}), 500

    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    file_uuid = str(uuid.uuid4())
    filename = f"{file_uuid}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Build/load model from ai.v2
        model = te.HCCTModel(te.CONFIG)
        model_path_candidate = os.path.join(PROJECT_ROOT, 'ai', 'best_model.pth')
        if os.path.exists(model_path_candidate):
            ckpt = torch.load(model_path_candidate, map_location=DEVICE)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
        else:
            rel_path = os.path.join(PROJECT_ROOT, 'ai', te.CONFIG.get('model_path', ''))
            if os.path.exists(rel_path):
                ckpt = torch.load(rel_path, map_location=DEVICE)
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                else:
                    model.load_state_dict(ckpt)
            else:
                raise FileNotFoundError('Trained model not found')

        model.to(DEVICE)
        model.eval()

        # Preprocess and inference
        img = te.enhanced_preprocessing_pipeline(filepath, te.CONFIG['image_size'], augment=False)
        img_tensor = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

        label_map = {0: 'CN', 1: 'AD'}

        # Explainability
        rollout = None
        thresholded_cam = None
        combined = None
        try:
            rollout = te.compute_attention_rollout_hybrid(model, img_tensor)
            gradcam_layer = model.cnn[4].conv
            gradcam = te.ThresholdedGradCAM3D(model, gradcam_layer)
            thresholded_cam, _ = gradcam.generate(img_tensor)
            if thresholded_cam is not None and rollout is not None:
                combined = te.combine_thresholded_gradcam_and_rollout(thresholded_cam, rollout)
        except Exception as ex:
            print('Explainability generation failed:', ex)

        # Save results directory
        session_dir = os.path.join(RESULTS_FOLDER, file_uuid)
        os.makedirs(session_dir, exist_ok=True)

        # We only save the final explainability image (no .npy files)

        # Try to create the full multi-view explainability image (grid) using ai.v2 helper
        explain_png_fname = None
        try:
            map_3d = None
            if combined is not None:
                map_3d = combined
            elif thresholded_cam is not None:
                map_3d = thresholded_cam
            elif rollout is not None:
                map_3d = rollout

            if map_3d is not None:
                explain_png_fname = f"{file_uuid}_explainability_grid.png"
                explain_png_path = os.path.join(session_dir, explain_png_fname)
                te.save_2d_slices_with_overlay(filepath, map_3d, explain_png_path, overlay_text=f"Pred: {label_map[pred_idx]} ({confidence:.2f})")
                # Ensure the saved PNG is written with no compression (lossless, compression level 0)
                try:
                    if os.path.exists(explain_png_path):
                        with Image.open(explain_png_path) as im:
                            im.save(explain_png_path, format='PNG', compress_level=0)
                except Exception as e:
                    print('Failed to re-save explainability PNG without compression:', e)
        except Exception as e:
            print('Failed to create full explainability image:', e)
            explain_png_fname = None

        # If explainability overlay wasn't created, save the central original slice (no modifications)
        if explain_png_fname is None:
            try:
                if 'img' in locals() and img is not None:
                    img_np2 = img.squeeze(0).cpu().numpy()
                else:
                    import nibabel as nib
                    nii = nib.load(filepath)
                    img_np2 = nii.get_fdata()
                    if img_np2.ndim > 3:
                        img_np2 = np.squeeze(img_np2)

                D = img_np2.shape[0]
                mid = D // 2
                axial_orig = np.rot90(img_np2[mid, :, :])
                def norm2(s):
                    s_min, s_max = s.min(), s.max()
                    return (s - s_min) / (s_max - s_min + 1e-8)
                axial_norm2 = norm2(axial_orig)
                explain_png_fname = f"{file_uuid}_original_slice.png"
                explain_png_path = os.path.join(session_dir, explain_png_fname)
                # Save as uncompressed PNG using Pillow
                try:
                    arr = (axial_norm2 * 255).astype(np.uint8)
                    im = Image.fromarray(arr)
                    if im.mode != 'L':
                        im = im.convert('L')
                    im.save(explain_png_path, format='PNG', compress_level=0)
                except Exception as e:
                    print('Failed to save fallback original slice as PNG:', e)
                    explain_png_fname = None
            except Exception:
                explain_png_fname = None

        return jsonify({
            'status': 'success',
            'data': {
                'prediction': label_map[pred_idx],
                'confidence': confidence,
                'uuid': file_uuid,
                'explainability_image_url': f"/results/{file_uuid}/{explain_png_fname}" if explain_png_fname else None,
            }
        })

    except Exception as e:
        print('Error in view.predict_explain:', e)
        return jsonify({'status': 'error', 'message': str(e)}), 500
