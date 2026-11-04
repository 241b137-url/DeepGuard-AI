
import os
from huggingface_hub import hf_hub_download
import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

# Model configuration
MODEL_NAME = "google/vit-base-patch16-224"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = None

def get_model_path():
    token = os.environ.get("HF_TOKEN")

    if not token:
        try:
            import streamlit as st
            token = st.secrets["HF_TOKEN"]
        except Exception:
            token = None

    if not token:
        raise RuntimeError("HF_TOKEN is missing from Streamlit Secrets.")

    return hf_hub_download(
        repo_id="madhur1230/deepguard-vit-model",
        filename="vit_deepfake_best.pth",
        repo_type="model",
        token=token
    )

# Face detector
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if FACE_CASCADE.empty():
    raise RuntimeError("Face detector could not be loaded.")

# Image preprocessing
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model():

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    model_path = get_model_path()

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


def predict_deepfake_video(video_path, model, num_frames=8):

    if not os.path.isfile(video_path):
        raise FileNotFoundError("Video not found.")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video contains no readable frames.")

    indices = np.linspace(
        0, total_frames - 1,
        min(num_frames, total_frames),
        dtype=int
    )

    tensors = []
    frame_results = []

    for index in indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = cap.read()

        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            continue

        x, y, w, h = max(
            faces,
            key=lambda box: box[2] * box[3]
        )

        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        tensors.append(TRANSFORM(Image.fromarray(face)))

        frame_results.append({
            "frame_number": int(index),
            "face_box": (int(x), int(y), int(w), int(h))
        })

    cap.release()

    if not tensors:
        return {
            "status": "NO_FACE",
            "message": "No detectable face found.",
            "frames_analyzed": 0
        }

    batch = torch.stack(tensors).to(DEVICE)

    with torch.inference_mode():
        logits = model(pixel_values=batch).logits
        probabilities = torch.softmax(logits, dim=1)

    probs = probabilities.cpu().numpy()

    fake_prob = float(probs[:, 0].mean())
    real_prob = float(probs[:, 1].mean())

    prediction = "FAKE" if fake_prob >= 0.5 else "REAL"

    for i, result in enumerate(frame_results):
        result["fake_probability"] = float(probs[i, 0])
        result["real_probability"] = float(probs[i, 1])

    return {
        "status": "SUCCESS",
        "prediction": prediction,
        "confidence": max(fake_prob, real_prob) * 100,
        "fake_probability": fake_prob * 100,
        "real_probability": real_prob * 100,
        "frames_analyzed": len(tensors),
        "frame_results": frame_results
    }
