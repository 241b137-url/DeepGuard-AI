
import streamlit as st
import os
import sys
import time
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from src.predictor import load_model, predict_deepfake_video

# Page configuration
st.set_page_config(
    page_title="DeepGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
    color: #172033;
}
.main-title {
    font-size: 44px;
    font-weight: 800;
    color: #174ea6;
}
.subtitle {
    color: #52647a;
    font-size: 17px;
}
.result-card {
    padding: 28px;
    border-radius: 16px;
    background: #ffffff;
    border: 1px solid #dce5f0;
    box-shadow: 0 4px 16px rgba(30, 60, 100, 0.07);
    color: #172033;
}
[data-testid="stMetric"] {
    background: #ffffff;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #dce5f0;
}
[data-testid="stMetricLabel"] {
    color: #52647a;
}
[data-testid="stMetricValue"] {
    color: #174ea6;
}
.stButton > button {
    background-color: #174ea6;
    color: white;
    border-radius: 10px;
    padding: 12px;
    border: none;
}
.stButton > button:hover {
    background-color: #123b80;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🛡️ DeepGuard AI</div>',
            unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">AI-Powered Deepfake Video Detection System</div>',
    unsafe_allow_html=True
)

st.divider()

# Load model once per app session
@st.cache_resource
def get_model():
    return load_model()

uploaded_file = st.file_uploader(
    "Upload a video for deepfake detection",
    type=["mp4", "avi", "mov", "mkv", "webm"]
)

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("🔍 Analyze Video", use_container_width=True):

        upload_dir = os.path.join(APP_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        video_path = os.path.join(upload_dir, uploaded_file.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with st.spinner("Loading model and analyzing video..."):

                model = get_model()

                start = time.time()

                result = predict_deepfake_video(
                    video_path,
                    model
                )

                processing_time = time.time() - start

            if result["status"] == "SUCCESS":

                st.success("Video analysis completed!")

                color = "#ef4444" if result["prediction"] == "FAKE" else "#22c55e"

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h2>Detection Result</h2>
                        <h1 style="color:{color};font-size:48px;">
                            {result['prediction']}
                        </h1>
                        <h3>Confidence: {result['confidence']:.2f}%</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Fake Probability",
                    f"{result['fake_probability']:.2f}%"
                )

                col2.metric(
                    "Real Probability",
                    f"{result['real_probability']:.2f}%"
                )

                col3.metric(
                    "Frames Analyzed",
                    result["frames_analyzed"]
                )

                st.metric(
                    "Processing Time",
                    f"{processing_time:.2f} seconds"
                )

                st.subheader("Frame-Level Analysis")

                frame_df = pd.DataFrame(result["frame_results"])
                st.dataframe(frame_df, use_container_width=True)

                st.caption(
                    "Predictions are model estimates, not definitive proof "
                    "of whether a video is authentic or manipulated."
                )

            else:
                st.warning(result["message"])

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
