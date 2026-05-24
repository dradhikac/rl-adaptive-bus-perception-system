import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Adaptive Autonomous Bus Perception System",
    layout="wide"
)

# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================================================
# CREATE OUTPUTS FOLDER
# =========================================================

os.makedirs("outputs", exist_ok=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("System Modules")

st.sidebar.write("✅ Object Detection")
st.sidebar.write("✅ Reliability Analysis")
st.sidebar.write("✅ Blind Spot Intelligence")
st.sidebar.write("✅ Adaptive Risk Intelligence")
st.sidebar.write("🔜 Sensor Fusion")
st.sidebar.write("🔜 Reinforcement Learning")

# =========================================================
# TITLE
# =========================================================

st.title(
    "Adaptive Autonomous Bus Perception System"
)

st.write(
    "AI-powered reliability, blind spot, and adaptive environmental intelligence dashboard."
)

# =========================================================
# FILE UPLOADERS
# =========================================================

uploaded_image = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

uploaded_video = st.file_uploader(
    "Upload Driving Video",
    type=["mp4"]
)

# =========================================================
# IMAGE PROCESSING
# =========================================================

if uploaded_image is not None:

    # -----------------------------------------------------
    # LOAD IMAGE
    # -----------------------------------------------------

    image = Image.open(uploaded_image)

    image_np = np.array(image)

    # -----------------------------------------------------
    # DISPLAY ORIGINAL IMAGE
    # -----------------------------------------------------

    st.subheader("Uploaded Image")

    st.image(
        image_np,
        use_container_width=True
    )

    # -----------------------------------------------------
    # YOLO OBJECT DETECTION
    # -----------------------------------------------------

    results = model(image_np)

    # Annotated image
    annotated_frame = results[0].plot()

    # -----------------------------------------------------
    # DISPLAY DETECTION RESULTS
    # -----------------------------------------------------

    st.subheader("Object Detection Results")

    st.image(
        annotated_frame,
        use_container_width=True
    )

    # -----------------------------------------------------
    # DETECTION DETAILS
    # -----------------------------------------------------

    st.subheader("Detection Information")

    object_count = len(results[0].boxes)

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        st.write(
            f"Object: {class_name} | Confidence: {confidence:.2f}"
        )

    # -----------------------------------------------------
    # ENVIRONMENTAL RELIABILITY
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY
    )

    # Brightness
    brightness = np.mean(gray)

    # Contrast
    contrast = gray.std()

    # Blur detection
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    blur_score = laplacian.var()

    # Reliability estimation
    reliability = 1.0

    if brightness < 60:
        reliability -= 0.3

    if contrast < 40:
        reliability -= 0.3

    if blur_score < 100:
        reliability -= 0.3

    reliability = max(
        0.0,
        min(reliability, 1.0)
    )

    # -----------------------------------------------------
    # ADAPTIVE RISK ANALYSIS
    # -----------------------------------------------------

    adaptive_risk = 0

    # Low reliability
    if reliability < 0.5:
        adaptive_risk += 3

    # Crowded environment
    if object_count > 10:
        adaptive_risk += 3

    elif object_count > 5:
        adaptive_risk += 2

    # -----------------------------------------------------
    # RISK LEVEL
    # -----------------------------------------------------

    risk_level = "LOW"

    if adaptive_risk >= 3:
        risk_level = "HIGH"

    if adaptive_risk >= 5:
        risk_level = "CRITICAL"

    # -----------------------------------------------------
    # METRIC CARDS
    # -----------------------------------------------------

    st.subheader("System Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Objects Detected",
        object_count
    )

    col2.metric(
        "Reliability Score",
        f"{reliability:.2f}"
    )

    col3.metric(
        "Risk Level",
        risk_level
    )

    # -----------------------------------------------------
    # ENVIRONMENTAL ANALYSIS
    # -----------------------------------------------------

    st.subheader("Environmental Intelligence")

    st.write(
        f"Brightness Score: {brightness:.2f}"
    )

    st.write(
        f"Contrast Score: {contrast:.2f}"
    )

    st.write(
        f"Blur Score: {blur_score:.2f}"
    )

    st.write(
        f"Camera Reliability Score: {reliability:.2f}"
    )

    # -----------------------------------------------------
    # ADAPTIVE RISK DISPLAY
    # -----------------------------------------------------

    st.subheader("Adaptive Risk Analysis")

    st.write(
        f"Adaptive Risk Score: {adaptive_risk}"
    )

    st.write(
        f"Adaptive Risk Level: {risk_level}"
    )

    # -----------------------------------------------------
    # ALERT SYSTEM
    # -----------------------------------------------------

    if risk_level == "LOW":

        st.success(
            "Environment appears safe."
        )

    elif risk_level == "HIGH":

        st.warning(
            "Elevated environmental risk detected."
        )

    elif risk_level == "CRITICAL":

        st.error(
            "CRITICAL danger detected!"
        )

    # -----------------------------------------------------
    # SAVE OUTPUT IMAGE
    # -----------------------------------------------------

    cv2.imwrite(
        "outputs/frontend_detection.jpg",
        cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_RGB2BGR
        )
    )

# =========================================================
# VIDEO SECTION
# =========================================================

if uploaded_video is not None:

    st.subheader("Uploaded Video")

    st.video(uploaded_video)

    st.info(
        "Full real-time video intelligence pipeline will be integrated in the next module."
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.write(
    "Adaptive Autonomous Bus Perception System | Module 1 Dashboard"
)