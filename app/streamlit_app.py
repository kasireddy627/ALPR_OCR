import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
from src.pipeline import ALPRPipeline
from src.preprocessor import ALPRPreprocessor

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="ALPR System",
    page_icon="🚗"
)

# ------------------------------------------------------------------
# Load components (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base, "config.yaml")
    return ALPRPipeline(config_path)

@st.cache_resource
def load_preprocessor():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base, "config.yaml")
    return ALPRPreprocessor(config_path)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🚗 ALPR System")
st.caption("YOLOv8 + Tesseract + SORT")

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("Settings")

source_mode = st.sidebar.radio(
    "Input Source",
    ["Upload Image", "Upload Video", "Webcam"],
    index=0
)

show_preprocessing = st.sidebar.checkbox(
    "Show Preprocessing (Actual Input to OCR)",
    value=False
)

if st.sidebar.button("Clear Database"):
    pipeline = load_pipeline()
    pipeline.database.clear_all()
    st.sidebar.success("Database cleared")

# ------------------------------------------------------------------
# Helper (FIXED — shows REAL preprocessing, not fake stages)
# ------------------------------------------------------------------
def show_preprocessing(cleaned):
    st.subheader("Preprocessed Image (Used for OCR)")
    # st.image(cleaned, use_container_width=True)
    st.image(cleaned, width='stretch')

# ------------------------------------------------------------------
# IMAGE MODE
# ------------------------------------------------------------------
if source_mode == "Upload Image":
    st.subheader("Image Mode")

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        pipeline = load_pipeline()
        preprocessor = load_preprocessor()

        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        annotated = pipeline.process_frame(frame, frame_number=1)

        col1, col2 = st.columns(2)

        with col1:
            st.caption("Original")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        with col2:
            st.caption("Annotated")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        detections = pipeline.detector.detect(frame)

        if detections:
            st.markdown("### OCR Results")

            for i, det in enumerate(detections):
                crop = det["crop"]

                cleaned = preprocessor.process(crop)
                result = pipeline.ocr.read_plate(cleaned)

                if result["text"]:
                    st.success(f"Plate {i+1}: {result['text']}")
                else:
                    st.warning(f"Plate {i+1}: Not detected")

                # FIX: show REAL input to OCR
                if show_preprocessing and i == 0:
                    show_preprocessing(cleaned)
        else:
            st.info("No plates detected")

        st.markdown("### Detection Log")
        reads = pipeline.database.get_all_reads()
        if reads:
            st.dataframe(reads, width='stretch')
        else:
            st.info("No records")

# ------------------------------------------------------------------
# VIDEO MODE
# ------------------------------------------------------------------
elif source_mode == "Upload Video":
    st.subheader("Video Mode")

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        pipeline = load_pipeline()

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.flush()

        video_placeholder = st.empty()
        stop_button = st.button("Stop")

        frame_count = 0
        fps_start = time.time()
        fps = 0

        for annotated in pipeline.run_on_video(tfile.name):
            if stop_button:
                break

            frame_count += 1

            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start
                fps = 30 / elapsed if elapsed > 0 else 0
                fps_start = time.time()

            video_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            )

        os.unlink(tfile.name)
        st.success("Done")

# ------------------------------------------------------------------
# WEBCAM MODE
# ------------------------------------------------------------------
elif source_mode == "Webcam":
    st.subheader("Webcam Mode")

    pipeline = load_pipeline()

    start = st.button("Start")
    stop = st.button("Stop")

    if start:
        cap = cv2.VideoCapture(0)

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = pipeline.process_frame(frame)

            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            )

        cap.release()
        st.success("Stopped")