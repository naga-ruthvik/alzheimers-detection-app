import os
import uuid
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import matplotlib.pyplot as plt
import numpy as np
import cv2
from model_utils import HCCTModel, GradCAM3D, preprocess_nifti_file
import sys

# Ensure project root is on sys.path so we can import the `ai` package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the ai explainability/test harness
try:
    from ai.v2 import test_explainability as te
except Exception:
    te = None

app = Flask(__name__)
# Allow large NIfTI files (up to 100MB)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
CORS(app, resources={r"/*": {"origins": "*"}}) 

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'research_code', 'models', 'best_model.pth')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "hidden_size": 512,
    "num_hidden_layers": 3,
    "num_attention_heads": 8,
    "intermediate_size": 1024,
    "num_classes": 2,
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load Model
model = HCCTModel(CONFIG).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

@app.route('/')
def index():
    return jsonify({
        'status': 'success',
        'message': "Alzheimer's Detection API is running",
        'endpoints': {
                'predict': '/predict (POST)',
                'predict_explain': '/predict_explain (POST) - uses ai package for explainability',
                'results': '/results/<filename> (GET)'
        }
    })

# Grad-CAM Setup
target_layer = model.cnn[-2].conv
gcam = GradCAM3D(model, target_layer)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    # Save uploaded file
    file_uuid = str(uuid.uuid4())
    filename = f"{file_uuid}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        print(f"--- Starting analysis for session {file_uuid} ---")
        
        # Preprocess and Predict
        print("Preprocessing NIfTI volume...")
        input_tensor, original_img = preprocess_nifti_file(filepath)
        input_tensor = input_tensor.to(DEVICE)
        
        print("Running model inference (CNN + Transformer)...")
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
        
        prediction_idx = np.argmax(probs)
        label_map = {0: 'CN', 1: 'AD'}
        prediction_name_map = {0: 'Cognitively Normal', 1: "Alzheimer's Disease"}
        
        print(f"Prediction complete: {label_map[prediction_idx]} ({probs[prediction_idx]:.2f})")
        
        # Save placeholder for explainability (will generate on demand or now)
        # For simplicity in this session, we generate it now
        print("Generating Grad-CAM heatmap...")
        cam, _ = gcam.generate(input_tensor)
        
        # Save Heatmap and Original Slices
        heatmap_filename = f"{file_uuid}_heatmap.png"
        original_filename = f"{file_uuid}_original.png"
        heatmap_path = os.path.join(RESULTS_FOLDER, heatmap_filename)
        original_path = os.path.join(RESULTS_FOLDER, original_filename)
        
        print("Generating and saving slices (Original & Heatmap)...")
        generate_slices(original_img, cam, original_path, heatmap_path)

        print(f"--- Session {file_uuid} completed successfully ---")

        return jsonify({
            'status': 'success',
            'data': {
                'prediction': label_map[prediction_idx],
                'predictionName': prediction_name_map[prediction_idx],
                'uuid': file_uuid,
                'confidence': float(probs[prediction_idx]),
                'heatmapUrl': f"/results/{heatmap_filename}",
                'originalUrl': f"/results/{original_filename}"
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/results/<path:filename>')
def get_result(filename):
    # Support nested paths like <uuid>/<file.png>
    return send_from_directory(RESULTS_FOLDER, filename)

def generate_slices(img, cam, original_path, heatmap_path):
    # Search for the slice with the MAXIMUM attention (the "hotspot")
    d, h, w = img.shape
    
    # Calculate max attention per slice
    slice_max_attention = [np.max(cam[i, :, :]) for i in range(d)]
    # Pick the best slice (index of max attention)
    best_idx = np.argmax(slice_max_attention)
    
    # If the max attention is too low, fall back to middle slice
    if slice_max_attention[best_idx] < 0.1:
        best_idx = d // 2
        
    print(f"Selecting slice {best_idx} with max attention {slice_max_attention[best_idx]:.4f}")
    
    # Extract the best slice
    axial_img = np.rot90(img[best_idx, :, :])
    axial_cam = np.rot90(cam[best_idx, :, :])
    
    # Normalize for display
    def norm(s):
        s_min, s_max = s.min(), s.max()
        return (s - s_min) / (s_max - s_min + 1e-8)
    
    axial_norm = norm(axial_img)
    
    # 1. Save high-res original
    plt.imsave(original_path, axial_norm, cmap='gray')
    
    # 2. Process Heatmap
    # Create tissue mask (where brain exists) - remove background air/noise
    # Thresholding slightly higher to ensure clean borders
    tissue_mask = axial_img > (axial_img.max() * 0.1)
    
    # Apply JET colormap for vibrant visualization
    heatmap = cv2.applyColorMap(np.uint8(255 * axial_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay logic: high-visibility
    gray_img_3ch = np.stack([axial_norm] * 3, axis=-1)
    
    # Filter CAM: only show if significant (>0.2) AND inside tissue
    # This prevents the "corner dots" seen previously
    cam_mask = (axial_cam > 0.2) & tissue_mask
    
    overlay = gray_img_3ch.copy()
    # Apply the heatmap with high transparency in attention zones
    overlay[cam_mask] = 0.2 * gray_img_3ch[cam_mask] + 0.8 * heatmap[cam_mask]
    
    plt.imsave(heatmap_path, overlay)


from view import explain_bp
app.register_blueprint(explain_bp)

# Add an alias route that calls the explain blueprint handler without disturbing existing endpoints
from view import predict_explain as predict_explain_view

@app.route('/predict_explain_v2', methods=['POST'])
def predict_explain_v2():
    return predict_explain_view()

if __name__ == '__main__':
    app.run(port=8080, debug=True)
