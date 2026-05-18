import os
import json
import time
import uuid
import base64
import ipaddress
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
YOLO_CONFIG_DIR = BASE_DIR / ".ultralytics"
YOLO_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

app = Flask(__name__, template_folder='.')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'static/uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
MODEL = None
MODEL_LOADED = False
SSL_DIR = BASE_DIR / "ssl"
SSL_CERT = SSL_DIR / "ricescan-local.crt"
SSL_KEY = SSL_DIR / "ricescan-local.key"

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    if response.content_type.startswith("text/html"):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
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
    global MODEL, MODEL_LOADED
    if MODEL_LOADED:
        return MODEL

    try:
        from ultralytics import YOLO
        model_path = BASE_DIR / "models" / "best.pt"
        if os.path.exists(model_path):
            MODEL = YOLO(model_path)
    except Exception as exc:
        app.logger.warning("Could not load YOLO model: %s", exc)
        MODEL = None

    MODEL_LOADED = True
    return MODEL

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

def get_lan_ip():
    """Best-effort LAN IP detection for HTTPS certificate SANs."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"

def ensure_https_certificate():
    """Create a local development certificate for browser camera access."""
    if SSL_CERT.exists() and SSL_KEY.exists():
        return SSL_CERT, SSL_KEY

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as exc:
        raise RuntimeError(
            "HTTPS needs cryptography. Run: pip install -r requirements.txt"
        ) from exc

    SSL_DIR.mkdir(exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    lan_ip = get_lan_ip()
    alt_names = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ]
    try:
        alt_names.append(x509.IPAddress(ipaddress.ip_address(lan_ip)))
    except ValueError:
        pass

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "BD"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "RiceScan Local"),
        x509.NameAttribute(NameOID.COMMON_NAME, "RiceScan Local Dev"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(days=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .sign(key, hashes.SHA256())
    )

    SSL_KEY.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    SSL_CERT.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return SSL_CERT, SSL_KEY

@app.route('/')
def index():
    if request.scheme == "http" and request.host.split(":")[0] not in {"localhost", "127.0.0.1"}:
        return redirect(f"https://{request.host.split(':')[0]}:5443/", code=302)
    return render_template('index.html', class_info=CLASS_INFO)

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

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
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    enable_https = os.environ.get("ENABLE_HTTPS", "0") == "1"
    port = int(os.environ.get("PORT", 5443 if enable_https else 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    ssl_context = None
    if enable_https:
        cert_path = Path(os.environ.get("SSL_CERT", SSL_CERT))
        key_path = Path(os.environ.get("SSL_KEY", SSL_KEY))
        if not cert_path.exists() or not key_path.exists():
            cert_path, key_path = ensure_https_certificate()
        ssl_context = (str(cert_path), str(key_path))
        print(f"HTTPS enabled. Open https://{get_lan_ip()}:{port}/ on your phone.")
    app.run(host=host, debug=debug, port=port, ssl_context=ssl_context)
