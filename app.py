import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify, render_template
from pathlib import Path

app = Flask(__name__)

# Configure paths
FIXED_FACE_PATH = Path("attached_assets/fixed_face.jpg")
OUTPUT_PATH = Path("output.jpg")

def load_fixed_face():
    if not FIXED_FACE_PATH.exists():
        raise FileNotFoundError(f"Fixed face image not found at {FIXED_FACE_PATH}")
    img = cv2.imread(str(FIXED_FACE_PATH))
    if img is None:
        raise ValueError(f"Failed to load image at {FIXED_FACE_PATH}")
    return img

try:
    fixed_face = load_fixed_face()
except Exception as e:
    print(f"Error loading fixed face: {e}")
    fixed_face = None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "fixed_face_loaded": fixed_face is not None})

@app.route('/swap', methods=['POST'])
def swap():
    if fixed_face is None:
        return jsonify({"error": "Fixed face not loaded"}), 500
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    try:
        image_file = request.files['image']
        image_np = np.frombuffer(image_file.read(), np.uint8)
        target_img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if target_img is None:
            return jsonify({"error": "Image decoding failed"}), 400
        swapped = cv2.resize(fixed_face, (target_img.shape[1], target_img.shape[0]))
        cv2.imwrite(str(OUTPUT_PATH), swapped)
        return send_file(str(OUTPUT_PATH), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate-art', methods=['POST'])
def generate_art():
    try:
        return send_file(str(FIXED_FACE_PATH), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
