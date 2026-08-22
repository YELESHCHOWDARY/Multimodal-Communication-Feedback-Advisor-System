import os
import tempfile
import threading
import webbrowser
import uuid
import json
import pickle
from io import BytesIO
from datetime import timedelta

from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from openai import audio
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv

import cv2
import math
import numpy as np
import speech_recognition as sr
import librosa
import soundfile as sf
import mediapipe as mp
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# Optional PDF / ReportLab + charting
REPORTLAB_OK = False
MATPLOTLIB_OK = False
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False


# ----------------- Load environment variables -----------------
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret')
MONGODB_URI = os.getenv('MONGODB_URI', None)
PORT = int(os.getenv('PORT', 5000))

# ----------------- JWT Setup -----------------
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-here')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)

# Explicit JWT header format (ADDED)
app.config['JWT_HEADER_NAME'] = 'Authorization'
app.config['JWT_HEADER_TYPE'] = 'Bearer'

# Allow token in header *and* query params (?token=...)
app.config['JWT_TOKEN_LOCATION'] = ['headers', 'query_string']
app.config['JWT_QUERY_STRING_NAME'] = 'token'

# Small clock leeway (ADDED)
app.config['JWT_DECODE_LEEWAY'] = 30

jwt = JWTManager(app)

# Allow Authorization/Content-Disposition headers (ADDED)
@app.after_request
def add_headers(resp):
    resp.headers.setdefault("Access-Control-Allow-Headers", "Authorization, Content-Type")
    resp.headers.setdefault("Access-Control-Expose-Headers", "Content-Disposition")
    return resp

# Silences Chrome devtools 404 spam (ADDED)
@app.get('/.well-known/appspecific/com.chrome.devtools.json')
def devtools_probe():
    return ("", 204)


# ----------------- JWT Error Handlers -----------------
@jwt.unauthorized_loader
def _unauthorized_loader(msg):
    return jsonify({"status": "error", "message": "Missing Authorization token"}), 401

@jwt.invalid_token_loader
def _invalid_token_loader(msg):
    return jsonify({"status": "error", "message": "Invalid token"}), 422

@jwt.expired_token_loader
def _expired_token_loader(jwt_header, jwt_payload):
    return jsonify({"status": "error", "message": "Token expired"}), 401

# ----------------- UPLOADS -----------------
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'webm', 'mp4', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------- MongoDB -----------------
if MONGODB_URI:
    try:
        client = MongoClient(MONGODB_URI)
        db = client.get_database()
        users_collection = db.users
        history_collection = db.history
        print("✅ Connected to MongoDB.")
    except Exception as e:
        print("❌ MongoDB connection failed:", e)
        users_collection = None
        history_collection = None
else:
    print("⚠️ No MONGODB_URI provided.")
    users_collection = None
    history_collection = None

# ----------------- FFmpeg Check -----------------
def ffmpeg_exists():
    from shutil import which
    return which("ffmpeg") is not None

HAS_FFMPEG = ffmpeg_exists()
if not HAS_FFMPEG:
    print("⚠️ ffmpeg not found — please install it")

# ----------------- Custom Emotion Model Loader -----------------
class CustomEmotionModel:
    """
    Loads a user-provided sklearn-like model (pickle) and predicts emotion probabilities.
    Env:
      EMOTION_MODEL_PATH: path to .pkl
      EMOTION_CLASSES: comma-separated class names (order must match model)
    """
    def __init__(self):
        self.model = None
        self.classes_ = ["neutral", "happy", "sad", "angry", "surprise"]
        path = os.getenv("EMOTION_MODEL_PATH", "").strip()
        classes_env = os.getenv("EMOTION_CLASSES", "").strip()

        if classes_env:
            self.classes_ = [c.strip() for c in classes_env.split(",") if c.strip()]

        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.model = pickle.load(f)
                print(f"✅ Loaded custom emotion model: {path}")
                print(f"✅ Classes: {self.classes_}")
            except Exception as e:
                print(f"❌ Failed to load emotion model: {e}. Falling back to heuristic.")
        else:
            print("ℹ️ No EMOTION_MODEL_PATH found or file missing. Using heuristic.")

    def predict_proba(self, X):
        """
        X: (n_samples, n_features) numpy array of aggregated features
        Returns: dict of {class_name: probability}
        """
        if self.model is not None:
            try:
                proba = self.model.predict_proba(X)[0]
                return {cls: float(p) for cls, p in zip(self.classes_, proba)}
            except Exception as e:
                print(f"❌ Model predict_proba failed: {e}. Falling back to heuristic.")
        # Heuristic fallback: derive a soft distribution from simple features
        # Expect X shape (1, 3): [mouth_ratio, eye_open, brow_raise]
        mr, eo, br = (X[0, 0], X[0, 1], X[0, 2])
        happy_score = max(0.0, min(1.0, 0.4 * (mr - 1.6) + 0.6 * br))  # smile+raised brows
        surprise_score = max(0.0, min(1.0, 0.6 * eo + 0.2 * (mr - 1.2)))
        angry_score = max(0.0, min(1.0, 0.6 * (1.2 - br)))
        sad_score = max(0.0, min(1.0, 0.5 * (1.1 - mr)))
        neutral_base = 1 - (happy_score + surprise_score + angry_score + sad_score) * 0.4
        neutral_score = max(0.05, min(1.0, neutral_base))

        raw = np.array([neutral_score, happy_score, sad_score, angry_score, surprise_score], dtype=float)
        raw = np.clip(raw, 0, None)
        if raw.sum() == 0:
            raw += 1e-6
        raw = raw / raw.sum()

        # Map to available classes
        dist = {"neutral": raw[0], "happy": raw[1], "sad": raw[2], "angry": raw[3], "surprise": raw[4]}
        # If custom classes differ, project shared names only and renormalize
        out = {k: dist.get(k, 0.0) for k in self.classes_}
        s = sum(out.values()) or 1.0
        for k in out:
            out[k] = float(out[k] / s)
        return out


face_emotion_model = CustomEmotionModel()

#----------------- HuBert Model Loader -----------------
# Labels (MUST match training)

LABELS = ["Excellent", "Good", "Moderate", "Poor"]

FLUENCY_WEIGHTS = {
    "Poor": 0.25,
    "Moderate": 0.50,
    "Good": 0.75,
    "Excellent": 1.00
}

# Load HuBERT

device ="cuda" if torch.cuda.is_available() else "cpu"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
hubert.eval()

# Fluency Classifier Architecture

class FluencyClassifier(nn.Module):
    def __init__(self, input_dim=1538, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
# Load pre-trained classifier

fluency_model = FluencyClassifier().to(device)
fluency_model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "best_fluency_model.pt"), map_location=device)
)
fluency_model.eval()
#  Core Prediction Function

def predict_fluency(audio_bytes):
    # -----------------------
    # Load audio
    # -----------------------
    audio, srate = sf.read(BytesIO(audio_bytes), dtype="float32")

    if srate != 16000:
        audio = librosa.resample(audio, srate, 16000)
        srate = 16000

    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    duration_sec = len(audio) / srate
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp: pcm_path = tmp.name
    sf.write(pcm_path, audio, samplerate=16000, subtype="PCM_16")

    # -----------------------
    # Speech Recognition (for WPM)
    # -----------------------
    recognizer = sr.Recognizer()
    text = ""
    try:
        with sr.AudioFile(pcm_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
    except Exception as e:
        print("Speech Recognition Failed")
        print("Error:", str(e))
        pass

    # -----------------------
    # Pace (WPM)
    # -----------------------
    words = len(text.split())
    duration_min = max(1e-6, duration_sec / 60)
    pace = round(words / duration_min, 2)

    # -----------------------
    # Speech validity gates
    # -----------------------

    if pace < 30 or words < 3:
        return {
            "fluency": "Poor",
            "pace": pace,
            "fluency_score": 0.0,
            "tone": 0.0,
            "message": "Insufficient speech detected. Please try again."
        }

    # -----------------------
    # HuBERT Feature Extraction
    # -----------------------
    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = hubert(**inputs).last_hidden_state

    emb = torch.cat([out.mean(1), out.std(1)], dim=1)

    # -----------------------
    # Feature Fusion
    # -----------------------
    def pace_score(pace):
        IDEAL = 140
        SIGMA = 30
        return math.exp(-((pace - IDEAL) ** 2) / (2 * SIGMA ** 2))

        
    word_confidence = min(1.0, words / 20)
    pace_effective = pace_score(pace) * word_confidence
    
    meta = torch.tensor([[pace_effective, duration_min]], dtype=torch.float32).to(device)
    features = torch.cat([emb, meta], dim=1)

    # -----------------------
    # Prediction
    # -----------------------
    with torch.no_grad():
        logits = fluency_model(features)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs))
    pred_label = LABELS[pred_idx]
    confidence = float(probs[pred_idx])

    # -----------------------
    # Fluency Score (Expected Value)
    # -----------------------
    fluency_score = sum(
        probs[i].item() * FLUENCY_WEIGHTS[LABELS[i]]
        for i in range(len(LABELS))
    )

    # -----------------------
    # Tone Score (Entropy-based)
    # -----------------------
    entropy = -sum(p.item() * math.log(p.item() + 1e-8) for p in probs)
    max_entropy = math.log(len(LABELS))
    tone_score = 1 - entropy / max_entropy

    return {
        "pace": pace,
        "fluency_score": round(fluency_score, 3),
        "tone": round(tone_score, 3)
    }

# ----------------- Utility: FaceMesh landmark indices -----------------
# Using MediaPipe face mesh: indexes for basic mouth/eye/brow metrics.
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

LEFT_BROW = 105
RIGHT_BROW = 334
LEFT_EYE_CENTER = 468  # iris center (approx)
RIGHT_EYE_CENTER = 473

def _dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def extract_face_features(landmarks, w, h):
    """
    landmarks: list of normalized (x,y) for the first detected face
    Returns: (mouth_ratio, eye_open, brow_raise)
    """
    pts = {}
    for idx in [MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM,
                LEFT_EYE_TOP, LEFT_EYE_BOTTOM, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                LEFT_BROW, RIGHT_BROW, LEFT_EYE_CENTER, RIGHT_EYE_CENTER]:
        lx = landmarks[idx].x * w
        ly = landmarks[idx].y * h
        pts[idx] = (lx, ly)

    mouth_w = _dist(pts[MOUTH_LEFT], pts[MOUTH_RIGHT]) + 1e-6
    mouth_h = _dist(pts[MOUTH_TOP], pts[MOUTH_BOTTOM]) + 1e-6
    mouth_ratio = mouth_w / mouth_h  # higher => smiling / open mouth

    left_eye_open = _dist(pts[LEFT_EYE_TOP], pts[LEFT_EYE_BOTTOM])
    right_eye_open = _dist(pts[RIGHT_EYE_TOP], pts[RIGHT_EYE_BOTTOM])
    eye_open = (left_eye_open + right_eye_open) / 2.0

    # Brow raise: distance between brow and eye center (avg both sides)
    lb = _dist(pts[LEFT_BROW], pts[LEFT_EYE_CENTER])
    rb = _dist(pts[RIGHT_BROW], pts[RIGHT_EYE_CENTER])
    brow_raise = (lb + rb) / 2.0

    # Normalize features to be roughly scale-invariant by face width (mouth_w)
    scale = mouth_w
    mouth_ratio_n = float(mouth_ratio)            # already a ratio
    eye_open_n = float(eye_open / (scale + 1e-6))
    brow_raise_n = float(brow_raise / (scale + 1e-6))

    # Clamp to reasonable ranges
    mouth_ratio_n = float(np.clip(mouth_ratio_n, 0.5, 3.5))
    eye_open_n = float(np.clip(eye_open_n, 0.05, 0.6))
    brow_raise_n = float(np.clip(brow_raise_n, 0.2, 1.5))

    return mouth_ratio_n, eye_open_n, brow_raise_n

# ----------------- Public Speaking Analyzer -----------------
class PublicSpeakingAnalyzer:
    @classmethod
    def load(cls):
        print("✅ PublicSpeakingAnalyzer loaded successfully.")
        return cls()

    def analyze(self, video_path):
        if not HAS_FFMPEG:
            return {"message": "ffmpeg not found on system."}

        uid = uuid.uuid4().hex
        converted = os.path.join(UPLOAD_FOLDER, f"converted_{uid}.mp4")
        wav_path = os.path.join(UPLOAD_FOLDER, f"audio_{uid}.wav")

        # Convert + Extract audio
        os.system(f'ffmpeg -y -i "{video_path}" -vcodec libx264 -acodec aac "{converted}" -loglevel error')
        
        if not os.path.exists(converted):
            return {"message": "Video conversion failed."}

        os.system(f'ffmpeg -y -i "{video_path}" -vn -ac 1 -ar 16000 -f wav "{wav_path}" -loglevel error')
        
        if not os.path.exists(wav_path):
            return {"message": "Audio extraction failed. Please upload a valid video."}

        cap = cv2.VideoCapture(converted)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps

        if duration < 10:
            cap.release()
            try:
                os.remove(converted)
                os.remove(wav_path)
            except: pass
            return {"message": "Video must be at least 10 seconds long."}

        # MediaPipe models
        with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as mp_face, \
        mp.solutions.hands.Hands() as mp_hands, \
        mp.solutions.pose.Pose() as mp_pose:

            face_frames = hand_frames = posture_frames = 0
            frame_count = 0

            # Emotion feature aggregation
            feat_sum = np.zeros(3, dtype=float)  # [mouth_ratio, eye_open, brow_raise]
            feat_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Face landmarks for eye contact + emotion features
                face_res = mp_face.process(rgb)
                if face_res.multi_face_landmarks:
                    face_frames += 1
                    landmarks = face_res.multi_face_landmarks[0].landmark
                    h, w = frame.shape[:2]
                    try:
                        mr, eo, br = extract_face_features(landmarks, w, h)
                        feat_sum += np.array([mr, eo, br], dtype=float)
                        feat_count += 1
                    except Exception:
                        pass

                if mp_hands.process(rgb).multi_hand_landmarks:
                    hand_frames += 1
                if mp_pose.process(rgb).pose_landmarks:
                    posture_frames += 1

            cap.release()


        eye_contact = round(face_frames / max(1, frame_count), 2)
        hand_gestures = round(hand_frames / max(1, frame_count), 2)
        posture = round(posture_frames / max(1, frame_count), 2)

        # Audio analysis (pace, fluency, tone proxy)
        with open(wav_path, "rb") as f: audio_bytes = f.read()
        results = predict_fluency(audio_bytes)
        pace = results["pace"]
        fluency = results["fluency_score"]
        tone = results["tone"]
        confidence = round((eye_contact + posture + hand_gestures) / 3, 2)

        # ---- Face-based emotion estimation (heuristic / optional ML) ----
        if feat_count > 0:
            feats = (feat_sum / feat_count).reshape(1, -1)
            proba = face_emotion_model.predict_proba(feats)
            happy = proba.get("happy", 0.0)
            surprise = proba.get("surprise", 0.0)
            sad = proba.get("sad", 0.0)
            angry = proba.get("angry", 0.0)
            emotion_score = happy + 0.5 * surprise - 0.35 * sad - 0.35 * angry
            emotion_score = float(np.clip(emotion_score, 0.0, 1.0))
        else:
            emotion_score = 0.5

        overall = round((confidence + fluency + tone * 0.8 + emotion_score * 0.8) / 3.6, 2)

        suggestions = []
        if eye_contact < 0.4: suggestions.append("Increase eye contact.")
        if posture < 0.5: suggestions.append("Improve posture.")
        if hand_gestures < 0.3: suggestions.append("Use more hand gestures.")
        if fluency < 0.5: suggestions.append("Improve fluency.")
        if emotion_score < 0.5: suggestions.append("Show more positive expression.")
        if pace > 180: suggestions.append("Speak slower.")
        if pace < 100: suggestions.append("Speak faster.")
        if not suggestions: suggestions.append("Great job!")

        return {
            "Eye Contact": eye_contact,
            "Posture": posture,
            "Hand Gestures": hand_gestures,
            "Emotion": round(emotion_score, 2),
            "Fluency": fluency,
            "Tone": tone,
            "Pace (WPM)": pace,
            "Confidence": confidence,
            "Overall Score": overall,
            "Suggestion": " ".join(suggestions)
        }

analyzer_model = PublicSpeakingAnalyzer.load()
# ----------------- ROUTES -----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/practice')
def practice():
    return render_template('practice.html')

@app.route('/history')
def history_page():
    return render_template('history.html')

# ✅ Serve uploaded videos
@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ✅ Direct video download (works with header OR ?token=)
@app.route('/download_video/<path:filename>')
@jwt_required()
def download_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

# ✅ JWT verify
@app.route('/verify_token')
@jwt_required()
def verify_token():
    username = get_jwt_identity()
    return jsonify({"status": "valid", "username": username})

# ✅ Analyze Route
@app.route('/analyze', methods=['POST'])
@jwt_required()
def analyze():
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video uploaded"}), 400

    video_file = request.files['video']
    if not video_file or not allowed_file(video_file.filename):
        return jsonify({"status": "error", "message": "Invalid file"}), 400

    filename = secure_filename(video_file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{filename}")
    video_file.save(save_path)

    try:
        results = analyzer_model.analyze(save_path)
        if "message" in results:
            return jsonify({"status": "error", "message": results["message"]}), 400
        
        if os.path.exists(save_path):
            os.remove(save_path)
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "success", "results": results})

# ✅ Save History
@app.route("/save_history", methods=["POST"])
@jwt_required()
def save_history():
    username = get_jwt_identity()

    if "video" not in request.files or "analysis_result" not in request.form:
        return jsonify({"message": "Missing data"}), 400

    video_file = request.files["video"]
    if not video_file or not allowed_file(video_file.filename):
        pass  # allow webm blob

    filename = f"{uuid.uuid4().hex}.webm"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(save_path)

    analysis_result = request.form["analysis_result"]
    try:
        analysis_data = json.loads(analysis_result)
    except Exception:
        analysis_data = analysis_result  # fallback to raw string

    history_collection.insert_one({
        "username": username,
        "video_path": filename,
        "analysis": analysis_data
    })

    return jsonify({"message": "Saved to history!"})

# ✅ History Data
@app.route("/get_history")
@jwt_required()
def get_history():
    username = get_jwt_identity()
    records = list(history_collection.find({"username": username}))
    for r in records:
        r["_id"] = str(r["_id"])
        if not isinstance(r.get("analysis"), str):
            try:
                r["analysis"] = json.dumps(r["analysis"])
            except Exception:
                r["analysis"] = str(r.get("analysis", ""))
    return jsonify(records)

# ✅ History stats for charts
@app.route("/history_stats")
@jwt_required()
def history_stats():
    username = get_jwt_identity()
    records = list(history_collection.find({"username": username}))
    if not records:
        return jsonify({"metrics": {}})

    sums = {
        "Eye Contact": 0.0,
        "Posture": 0.0,
        "Hand Gestures": 0.0,
        "Emotion": 0.0,
        "Fluency": 0.0,
        "Tone": 0.0,
        "Confidence": 0.0,
        "Overall Score": 0.0,
    }
    pace_values = []
    cnt = 0

    for r in records:
        a = r.get("analysis")
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except Exception:
                a = {}
        if not isinstance(a, dict):
            continue

        cnt += 1
        for k in sums.keys():
            v = a.get(k)
            if isinstance(v, (int, float)):
                sums[k] += float(v)
        p = a.get("Pace (WPM)")
        if isinstance(p, (int, float)):
            pace_values.append(float(p))

    if cnt == 0:
        return jsonify({"metrics": {}})

    metrics_pct = {k: round((sums[k]/cnt) * 100, 2) for k in sums.keys()}
    avg_pace = round(sum(pace_values)/len(pace_values), 2) if pace_values else 0.0

    return jsonify({"metrics": metrics_pct, "avg_pace": avg_pace})

# ✅ Export single analysis as PDF (INLINE display)
@app.route("/export_analysis_pdf/<id>")
@jwt_required()
def export_analysis_pdf(id):
    if not REPORTLAB_OK:
        # Keep returning PDF to avoid browser "failed to load" — build a minimal PDF message
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, 800, "PDF export: reportlab not installed on server.")
        c.showPage()
        c.save()
        buffer.seek(0)
        return send_file(buffer, mimetype="application/pdf", as_attachment=False, download_name="error.pdf")

    username = get_jwt_identity()
    try:
        doc = history_collection.find_one({"_id": ObjectId(id), "username": username})
    except Exception:
        doc = None
    if not doc:
        # Return a PDF error page for inline display to avoid browser error screen
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, 800, "Record not found")
        c.showPage()
        c.save()
        buffer.seek(0)
        return send_file(buffer, mimetype="application/pdf", as_attachment=False, download_name="notfound.pdf")

    analysis = doc.get("analysis")
    if isinstance(analysis, str):
        try:
            analysis = json.loads(analysis)
        except Exception:
            pass

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Communication Practice - Analysis Summary")
    y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Username: {username}")
    y -= 18
    c.drawString(40, y, f"Record ID: {str(doc.get('_id'))}")
    y -= 18
    c.drawString(40, y, f"Video file: {doc.get('video_path', '-')}")
    y -= 24

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Metrics")
    y -= 18
    c.setFont("Helvetica", 11)

    def line(text):
        nonlocal y
        if y < 60:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 11)
        c.drawString(50, y, text)
        y -= 16

    if isinstance(analysis, dict):
        for k in ["Eye Contact","Posture","Hand Gestures","Emotion","Fluency","Tone","Confidence","Overall Score","Pace (WPM)"]:
            v = analysis.get(k, "-")
            if isinstance(v, (int, float)) and k != "Pace (WPM)":
                v = f"{round(v*100, 1)}%"
            line(f"{k}: {v}")
        line(f"Suggestion: {analysis.get('Suggestion', '-')}")
    else:
        line("Analysis:")
        line(str(analysis))

    c.showPage()
    c.save()
    buffer.seek(0)

    # INLINE display
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=False,
        download_name=f"analysis_{str(doc.get('_id'))}.pdf"
    )

@app.route("/export_overall_pdf")
@jwt_required()
def export_overall_pdf():
    username = get_jwt_identity()

    try:
        records = list(history_collection.find({"username": username}))
    except:
        records = []

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    LEFT, RIGHT = 40, 40
    TOP, BOTTOM = 60, 60
    MAX_W = width - LEFT - RIGHT

    # Updated spacing for labels
    LABEL_COL = LEFT
    VALUE_COL = LEFT + 90  # ✅ tightened spacing

    def safe_line_break(px):
        nonlocal y
        if y - px < BOTTOM:
            c.showPage()
            y = height - TOP

    y = height - TOP

    # ------------------------------------------------------
    # ✅ HEADER TITLE
    # ------------------------------------------------------
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(width / 2, y, "Overall Performance Report")
    y -= 48

    # ------------------------------------------------------
    # ✅ USER (clean spacing)
    # ------------------------------------------------------
    c.setFont("Helvetica-Bold", 14)
    c.drawString(LABEL_COL, y, "User:")
    c.setFont("Helvetica", 14)
    c.drawString(VALUE_COL, y, username)
    y -= 24

    # ------------------------------------------------------
    # ✅ SESSIONS (clean spacing)
    # ------------------------------------------------------
    c.setFont("Helvetica-Bold", 14)
    c.drawString(LABEL_COL, y, "Sessions:")
    c.setFont("Helvetica", 14)
    c.drawString(VALUE_COL, y, str(len(records)))
    y -= 30

    # Divider Line
    c.setStrokeColorRGB(0.70, 0.70, 0.70)
    c.setLineWidth(1.2)
    c.line(LEFT, y, width - RIGHT, y)
    y -= 40

    if not records:
        c.setFont("Helvetica", 14)
        c.drawString(LEFT, y, "No history available.")
        c.showPage()
        c.save()
        buffer.seek(0)
        return send_file(buffer, mimetype="application/pdf", as_attachment=False)

    # ------------------------------------------------------
    # ✅ Compute averages
    # ------------------------------------------------------
    metric_keys = [
        "Eye Contact", "Posture", "Hand Gestures",
         "Fluency", "Tone",
        "Confidence", "Overall Score"
    ]

    sums = {k: 0 for k in metric_keys}
    pace_values = []
    count = 0

    for entry in records:
        try:
            a = entry["analysis"]
            if isinstance(a, str):
                a = json.loads(a)
        except:
            continue

        count += 1
        for k in metric_keys:
            if isinstance(a.get(k), (float, int)):
                sums[k] += a[k]

        if isinstance(a.get("Pace (WPM)"), (float, int)):
            pace_values.append(a["Pace (WPM)"])

    if count == 0:
        count = 1

    averages = {k: round((sums[k] / count) * 100, 2) for k in metric_keys}
    avg_pace = round(sum(pace_values)/len(pace_values), 2) if pace_values else 0

    # ------------------------------------------------------
    # ✅ LARGE BAR CHART
    # ------------------------------------------------------
    chart_img = None
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(8.3, 5), dpi=160)

        labels = metric_keys
        values = [averages[k] for k in labels]

        x = np.arange(len(labels))
        bars = ax.bar(x, values, width=0.40, color="#4A68F0")

        for xi, bar, v in zip(x, bars, values):
            ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")

        ax.set_ylim(0, max(values) + 20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha='right')
        ax.set_ylabel("Percent (%)")
        ax.set_title("Overall Performance Metrics", fontsize=14)

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        chart_img = ImageReader(buf)

    except:
        chart_img = None

    # Draw chart
    if chart_img:
        chart_w = MAX_W
        chart_h = 260

        safe_line_break(chart_h)
        c.drawImage(chart_img, LEFT, y - chart_h, width=chart_w, height=chart_h)
        y -= chart_h + 40
    else:
        c.setFont("Helvetica-Oblique", 12)
        c.drawString(LEFT, y, "(Chart unavailable)")
        y -= 20

    # ------------------------------------------------------
    # ✅ BIGGER TABLE (center aligned)
    # ------------------------------------------------------
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors

    table_data = [["Metric", "Average (%)"]]
    for k in metric_keys:
        table_data.append([k, str(averages[k])])
    table_data.append(["Pace (WPM)", str(avg_pace)])

    table = Table(
        table_data,
        colWidths=[MAX_W * 0.6, MAX_W * 0.4],
        rowHeights=26
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 13),

        # ✅ FULL CENTER ALIGNMENT
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),

        ("GRID", (0,0), (-1,-1), 0.8, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
    ]))

    tw, th = table.wrap(0, 0)
    safe_line_break(th)
    table.drawOn(c, LEFT, y - th)
    y -= th + 30

    # Finish PDF
    c.showPage()
    c.save()
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=False,
        download_name="overall_performance.pdf"
    )

# ✅ Delete one history item
@app.route("/delete_history/<id>", methods=["DELETE"])
@jwt_required()
def delete_history(id):
    username = get_jwt_identity()
    try:
        history_collection.delete_one({"_id": ObjectId(id), "username": username})
        return jsonify({"message": "Deleted"}), 200
    except Exception as e:
        return jsonify({"message": f"Delete failed: {e}"}), 400

# ✅ Clear all history
@app.route("/clear_history", methods=["DELETE"])
@jwt_required()
def clear_history():
    username = get_jwt_identity()
    history_collection.delete_many({"username": username})
    return jsonify({"message": "All cleared"}), 200

# ✅ Register
@app.route('/register_user', methods=['POST'])
def register_user():
    username = request.form.get('username')
    email = request.form.get('email', '').lower()
    password = request.form.get('password')

    if not username or not email or not password:
        return jsonify({"status": "error", "message": "All fields required"}), 400

    if users_collection.find_one({'email': email}):
        return jsonify({"status": "error", "message": "Email exists"}), 400

    hashed_pw = generate_password_hash(password)
    users_collection.insert_one({
        "username": username,
        "email": email,
        "password": hashed_pw
    })

    token = create_access_token(identity=username)
    return jsonify({"status": "success", "username": username, "access_token": token,"redirect":"/home"})

# ✅ Login (JSON or form)
@app.route('/login_user', methods=['POST'])
def login_user():
    if request.is_json:
        data = request.get_json()
        email = (data.get('email') or '').lower()
        password = (data.get('password') or '')
    else:
        email = (request.form.get('email') or '').lower()
        password = (request.form.get('password') or '')

    user = users_collection.find_one({"email": email})
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"status": "error", "message": "Invalid login"}), 401

    token = create_access_token(identity=user["username"])
    return jsonify({
        "status": "success",
        "username": user["username"],
        "access_token": token,
        "redirect": "/home"
    })

# ✅ User Info
@app.route('/user_info')
@jwt_required()
def user_info():
    return jsonify({"username": get_jwt_identity()})

# ----------------- Auto Launch -----------------
def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        threading.Timer(1.5, open_browser).start()
    app.run(port=PORT, debug=True)
