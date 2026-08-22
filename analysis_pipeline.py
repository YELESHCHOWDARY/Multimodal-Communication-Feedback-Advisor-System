# analysis_pipeline.py
import os
import subprocess
import tempfile
import json
import math
import cv2
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import detect_silence
import librosa
from textblob import TextBlob

try:
    from transformers import pipeline
    _EMO_MODEL_AVAILABLE = True
except Exception:
    _EMO_MODEL_AVAILABLE = False


def _run_ffmpeg_extract_audio(video_path: str, out_wav: str):
    print("🎧 Extracting audio from video...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"✅ Audio saved to {out_wav}")


def _run_ffmpeg_convert_h264(video_path: str, out_mp4: str):
    print("🎞️ Converting video to H.264 for OpenCV compatibility...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vcodec", "libx264", "-acodec", "aac", out_mp4
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"✅ Video converted to {out_mp4}")


def _transcribe_audio(audio_wav):
    print("🗣️ Transcribing speech from audio...")
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_wav) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        print("✅ Speech transcription successful.")
        return text
    except Exception:
        print("⚠️ Speech transcription failed.")
        return ""


def analyze_video(video_path: str, use_emotion_model: bool = True):
    print(f"\n🚀 Starting analysis for: {video_path}")
    tmpdir = tempfile.mkdtemp(prefix="psa_")
    converted = os.path.join(tmpdir, "converted.mp4")
    wav_path = os.path.join(tmpdir, "audio.wav")

    # Convert and extract audio
    try:
        _run_ffmpeg_convert_h264(video_path, converted)
    except subprocess.CalledProcessError:
        print("⚠️ ffmpeg conversion failed, using original video.")
        converted = video_path

    try:
        _run_ffmpeg_extract_audio(converted, wav_path)
    except subprocess.CalledProcessError:
        print("⚠️ Could not extract audio.")
        open(wav_path, "wb").close()

    audio_present = os.path.exists(wav_path) and os.path.getsize(wav_path) > 100

    # ---------------- VIDEO ANALYSIS ----------------
    print("\n📹 Starting Video Analysis...")
    import mediapipe as mp
    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(converted)
    if not cap.isOpened():
        print("⚠️ Could not open video file.")
        return {"error": "Could not open video file"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"Total frames detected: {total_frames}, FPS: {fps}")

    eye_frames = hand_frames = posture_frames = 0
    gesture_motion_sum = 0.0
    prev_hand_centers = None
    processed_frames = 0

    with mp_face.FaceMesh(refine_landmarks=True) as face, \
         mp_hands.Hands(max_num_hands=2) as hands, \
         mp_pose.Pose() as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frames += 1
            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            fres = face.process(rgb)
            if fres.multi_face_landmarks:
                eye_frames += 1

            hres = hands.process(rgb)
            if hres.multi_hand_landmarks:
                hand_frames += 1
                centers = []
                for hl in hres.multi_hand_landmarks:
                    xs = [lm.x for lm in hl.landmark]
                    ys = [lm.y for lm in hl.landmark]
                    centers.append((np.mean(xs), np.mean(ys)))
                if centers:
                    arr = np.array(centers)
                    if prev_hand_centers is not None and arr.shape == prev_hand_centers.shape:
                        gesture_motion_sum += float(np.linalg.norm(arr - prev_hand_centers))
                    prev_hand_centers = arr
            else:
                prev_hand_centers = None

            pres = pose.process(rgb)
            if pres.pose_landmarks:
                lm = pres.pose_landmarks.landmark
                L_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                R_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                if abs(L_sh.y - R_sh.y) < 0.08:
                    posture_frames += 1

    cap.release()

    eye_score = round(eye_frames / total_frames, 3)
    hand_score = round(hand_frames / total_frames, 3)
    posture_score = round(posture_frames / total_frames, 3)
    gesture_energy = round(min(gesture_motion_sum / max(1, processed_frames), 1.0), 3)

    print(f"✅ Video Analysis Done! Eye: {eye_score}, Hand: {hand_score}, Posture: {posture_score}")

    # ---------------- AUDIO ANALYSIS ----------------
    text = ""
    fluency_score = pronunciation_score = pace_wpm = 0.0
    tone_confidence = None
    dominant_emotion = None
    confidence_score = None
    sentiment = None

    if audio_present:
        text = _transcribe_audio(wav_path)
        words = text.split()
        word_count = len(words)
        print(f"🗒️ Words detected: {word_count}")

        audio = AudioSegment.from_wav(wav_path)
        pauses = detect_silence(audio, min_silence_len=400, silence_thresh=audio.dBFS - 14)
        total_pause = sum(p[1] - p[0] for p in pauses)
        pause_ratio = total_pause / len(audio)
        pace_wpm = word_count / (len(audio) / (1000 * 60))
        fluency_score = round(min((1 - pause_ratio) * (pace_wpm / 150), 1.0), 3)
        pronunciation_score = round(min(pace_wpm / 160, 1.0), 3)

        if use_emotion_model and _EMO_MODEL_AVAILABLE:
            print("🎭 Running emotion analysis...")
            emo = pipeline("audio-classification", model="superb/hubert-large-superb-er")
            res = emo(wav_path)
            dominant_emotion = res[0]["label"]
            tone_confidence = round(res[0]["score"], 3)
            print(f"Emotion: {dominant_emotion}, Confidence: {tone_confidence}")

        y, sr_ = librosa.load(wav_path, sr=None)
        energy = np.mean(librosa.feature.rms(y=y))
        confidence_score = round(min(energy * 20 + (tone_confidence or 0) * 0.8, 1.0), 3)
        sentiment = round(TextBlob(text).sentiment.polarity, 3)

    print("✅ Audio Analysis Done!")

    report = {
        "Eye Contact": eye_score,
        "Hand Gestures": hand_score,
        "Gesture Energy": gesture_energy,
        "Posture": posture_score,
        "Fluency": fluency_score,
        "Pronunciation": pronunciation_score,
        "Pace (WPM)": round(pace_wpm, 2),
        "Tone Confidence": tone_confidence,
        "Dominant Emotion": dominant_emotion,
        "Confidence": confidence_score,
        "Sentiment": sentiment,
    }

    print("\n📊 FINAL REPORT:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    return report