import os
import json
import time
import uuid
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
YOLO_CONFIG_DIR = BASE_DIR / ".ultralytics"
YOLO_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

app = Flask(__name__, template_folder='.')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'static/uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# Class metadata for rice false smut stages
CLASS_INFO = {
    "initial": {
        "label": "Initial Stage",
        "color": "#22c55e",
        "description": "Early infection — spore balls just forming, yellowish-green appearance.",
        "risk": "Low",
        "action": "Monitor closely. Consider preventive fungicide at boot stage."
    },
    "yellow": {
        "label": "Yellow Stage",
        "color": "#eab308",
        "description": "Spore balls fully yellow, actively sporulating.",
        "risk": "Medium",
        "action": "Apply targeted fungicide. Mark affected area for harvest monitoring."
    },
    "transition": {
        "label": "Transition Stage",
        "color": "#f97316",
        "description": "Spore balls transitioning from yellow to dark olive-green.",
        "risk": "High",
        "action": "Immediate fungicide application. Avoid harvesting from contaminated rows."
    },
    "last": {
        "label": "Last Stage",
        "color": "#ef4444",
        "description": "Dark olive to black spore balls — maximum sporulation, seeds destroyed.",
        "risk": "Critical",
        "action": "Do not use seeds from this batch. Deep plough after harvest. Rotate crops."
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def try_load_model():
    """Attempt to load a YOLO model if ultralytics is available."""
    try:
        from ultralytics import YOLO
        model_path = BASE_DIR / "models" / "best.pt"
        if os.path.exists(model_path):
            return YOLO(model_path)
    except Exception as exc:
        app.logger.warning("Could not load YOLO model: %s", exc)
        pass
    return None

def run_inference(image_path):
    """
    Run YOLO inference if model exists, otherwise return demo results.
    Returns list of detections: [{class, confidence, bbox, ...}, ...]
    """
    model = try_load_model()
    
    if model is not None:
        results = model(image_path, conf=0.1)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                    "info": CLASS_INFO.get(cls_name, {})
                })
        return detections, True
    else:
        # Demo mode — simulate detections
        import random
        random.seed(42)
        classes = list(CLASS_INFO.keys())
        n = random.randint(1, 4)
        detections = []
        for _ in range(n):
            cls = random.choice(classes)
            detections.append({
                "class": cls,
                "confidence": round(random.uniform(0.55, 0.97), 3),
                "bbox": [
                    random.randint(50, 200),
                    random.randint(50, 200),
                    random.randint(250, 500),
                    random.randint(250, 500)
                ],
                "info": CLASS_INFO[cls]
            })
        return detections, False

def compute_summary(detections):
    """Aggregate stats from detections."""
    if not detections:
        return {"total": 0, "dominant": None, "avg_conf": 0, "class_counts": {}}
    
    class_counts = {}
    total_conf = 0
    for d in detections:
        c = d["class"]
        class_counts[c] = class_counts.get(c, 0) + 1
        total_conf += d["confidence"]
    
    dominant = max(class_counts, key=class_counts.get)
    return {
        "total": len(detections),
        "dominant": dominant,
        "dominant_info": CLASS_INFO.get(dominant, {}),
        "avg_conf": round(total_conf / len(detections), 3),
        "class_counts": class_counts
    }

@app.route('/')
def index():
    return render_template('index.html', class_info=CLASS_INFO)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not supported"}), 400
    
    # Save upload
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)
    
    # Run inference
    start = time.time()
    detections, model_loaded = run_inference(filepath)
    elapsed = round(time.time() - start, 3)
    
    summary = compute_summary(detections)
    
    return jsonify({
        "success": True,
        "image_url": f"/static/uploads/{filename}",
        "model": "best.pt",
        "model_loaded": model_loaded,
        "inference_time": elapsed,
        "detections": detections,
        "summary": summary
    })

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True, port=5000)
