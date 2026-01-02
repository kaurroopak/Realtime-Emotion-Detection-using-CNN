import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Real-Time Mood Detection",
    page_icon="😊",
    layout="centered"
)

# -----------------------------
# CONSTANTS
# -----------------------------
LABELS = ['Happy', 'Neutral', 'Sad']
IMG_SIZE = (224, 224)
MODEL_PATH = "mobilenetv2_mood_3class.tflite"

# -----------------------------
# LOAD TFLITE MODEL
# -----------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(image_np):
    img = cv2.resize(image_np, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])[0]
    latency = (time.time() - start) * 1000

    return probs, latency

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    "<h1 style='text-align:center;'>😊 Real-Time Mood Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Detects <b>Happy</b>, <b>Neutral</b>, or <b>Sad</b> emotions using facial images</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("📌 Project Info")
st.sidebar.write("""
**Model:** MobileNetV2 (CNN)  
**Classes:** Happy, Neutral, Sad  
**Deployment:** TensorFlow Lite  
**Input:** Image / Webcam  
""")

st.sidebar.markdown("### 🧪 AI Exploration")
apply_blur = st.sidebar.checkbox("Apply Gaussian Blur (Robustness Test)")

# -----------------------------
# INPUT MODE
# -----------------------------
mode = st.radio(
    "Choose Input Method",
    ["📷 Webcam", "🖼️ Image Upload"],
    horizontal=True
)

# -----------------------------
# IMAGE INPUT
# -----------------------------
image_np = None

if mode == "🖼️ Image Upload":
    uploaded = st.file_uploader("Upload a facial image", type=["jpg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="Original Input Image", width=300)

elif mode == "📷 Webcam":
    cam_image = st.camera_input("Capture image")
    if cam_image:
        image = Image.open(cam_image).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="Original Captured Image", width=300)

# -----------------------------
# PREDICTION OUTPUT
# -----------------------------
if image_np is not None:
    test_image = image_np.copy()

    # -------- MODULE 6: CONDITION MODIFICATION --------
    if apply_blur:
        test_image = cv2.GaussianBlur(test_image, (9, 9), 0)
        st.image(test_image, caption="Modified Image (Gaussian Blur Applied)", width=300)

    probs, latency = predict(test_image)
    pred_idx = np.argmax(probs)
    mood = LABELS[pred_idx]

    st.markdown("### 🎯 Prediction Result")

    st.success(f"**Detected Mood: {mood}**")

    # Confidence chart
    df = pd.DataFrame({
        "Mood": LABELS,
        "Confidence": probs
    })
    st.bar_chart(df.set_index("Mood"))

    st.info(f"⏱️ Inference Time: {latency:.2f} ms")

    if apply_blur:
        st.warning(
            "⚠️ Confidence may decrease due to image blur. "
            "This demonstrates the model’s sensitivity to spatial feature degradation."
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:13px;'>ELC Project – 5th Semester | Real-Time Data Analysis</p>",
    unsafe_allow_html=True
)
