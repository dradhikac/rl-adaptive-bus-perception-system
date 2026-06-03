import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sys
import os

from PIL import Image


# Add project root folder to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.autonomous_brain import analyze_scene


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Autonomous Bus AI Dashboard",
    page_icon="🚍",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title(
    "🚍 Autonomous Bus Perception & Decision System"
)

st.markdown(
    """
    Intelligent perception, sensor fusion,
    risk assessment and autonomous decision engine.
    """
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("System Modules")

st.sidebar.success("✅ Reliability Intelligence")

st.sidebar.success("✅ Sensor Fusion")

st.sidebar.success("✅ Traffic Intelligence")

st.sidebar.success("✅ Risk Intelligence")

st.sidebar.success("✅ Decision Engine")

st.sidebar.info("🔄 Video Analytics")

st.sidebar.info("🔜 Blind Spot Intelligence")

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
# IMAGE ANALYSIS
# =========================================================

if uploaded_image is not None:

    image = Image.open(uploaded_image)

    image_np = np.array(image)

    result = analyze_scene(image_np)

    # =====================================================
    # OVERVIEW METRICS
    # =====================================================

    st.subheader("🚍 Autonomous Driving Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Objects",
        result["object_count"]
    )

    col2.metric(
        "Signal",
        result["traffic_signal"]
    )

    col3.metric(
        "Risk",
        result["risk_score"]
    )

    col4.metric(
        "Decision",
        result["decision"]
    )

    col5.metric(
        "Speed",
        f'{result["recommended_speed"]} km/h'
    )

    # =====================================================
    # TABS
    # =====================================================

    tabs = st.tabs([
    "🚍 Perception",
    "📷 Reliability",
    "📡 Fusion",
    "🚦 Traffic",
    "🚗 Distance",
    "⚠️ Risk",
    "🤖 Decision",
    "🎥 Video Analytics"
])
    # =====================================================
    # TAB 1 : PERCEPTION
    # =====================================================

    with tabs[0]:

        st.subheader(
            "Detected Scene"
        )

        st.image(
            result["annotated_image"],
            use_container_width=True
        )

        st.write(
            f'Objects: {result["object_count"]}'
        )

        st.write(
            f'Vehicles: {result["vehicle_count"]}'
        )

        st.write(
            f'Pedestrians: {result["pedestrian_count"]}'
        )

    # =====================================================
    # TAB 2 : RELIABILITY
    # =====================================================

    with tabs[1]:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Brightness",
            result["brightness"]
        )

        c2.metric(
            "Contrast",
            result["contrast"]
        )

        c3.metric(
            "Blur",
            result["blur_score"]
        )

        c4.metric(
            "Reliability",
            result["camera_reliability"]
        )

    # =====================================================
    # TAB 3 : SENSOR FUSION
    # =====================================================

    with tabs[2]:

        st.metric(
            "Fusion Confidence",
            result["fusion_confidence"]
        )

        st.progress(
            float(result["fusion_confidence"])
        )

        st.write(
            "Camera Reliability"
        )

        st.write(
            result["camera_reliability"]
        )

        st.write(
            "LiDAR Reliability"
        )

        st.write(
            "1.0"
        )

    # =====================================================
    # TAB 4 : TRAFFIC
    # =====================================================

    with tabs[3]:

        signal = result["traffic_signal"]

        if signal == "RED":

            st.error(
                "🔴 RED SIGNAL"
            )

        elif signal == "YELLOW":

            st.warning(
                "🟡 YELLOW SIGNAL"
            )

        elif signal == "GREEN":

            st.success(
                "🟢 GREEN SIGNAL"
            )

        else:

            st.info(
                "⚪ SIGNAL UNKNOWN"
            )

    # =====================================================
    # TAB 5 : DISTANCE
    # =====================================================

    with tabs[4]:

        st.metric(
            "Lead Vehicle Distance",
            result["vehicle_distance"]
        )

    # =====================================================
    # TAB 6 : RISK
    # =====================================================

    with tabs[5]:

        st.metric(
            "Risk Score",
            result["risk_score"]
        )

        st.metric(
            "Risk Level",
            result["risk_level"]
        )

        st.subheader(
            "Reasons"
        )

        for reason in result["reasons"]:

            st.write(
                f"• {reason}"
            )

    # =====================================================
    # TAB 7 : DECISION
    # =====================================================

    with tabs[6]:

        decision = result["decision"]

        speed = result["recommended_speed"]

        if decision == "STOP":

            st.error(
                f"🛑 STOP\n\nRecommended Speed: {speed} km/h"
            )

        elif decision == "SLOW DOWN":

            st.warning(
                f"⚠️ SLOW DOWN\n\nRecommended Speed: {speed} km/h"
            )

        else:

            st.success(
                f"✅ MOVE\n\nRecommended Speed: {speed} km/h"
            )

# =========================================================
# VIDEO SECTION
# =========================================================

if uploaded_video is not None:

    st.subheader(
        "Uploaded Driving Video"
    )

    st.video(
        uploaded_video
    )

    st.info(
        "Video Timeline Analytics will be connected next."
    )
    
    with tabs[7]:

        st.subheader(
        "Video Risk Analytics"
    )

    try:

        df = pd.read_csv(
            "outputs/video_risk_timeline.csv"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        st.subheader(
            "Risk Score Trend"
        )

        st.line_chart(
            df.set_index("time_sec")[
                "risk_score"
            ]
        )

        st.subheader(
            "Recommended Speed Trend"
        )

        st.line_chart(
            df.set_index("time_sec")[
                "recommended_speed"
            ]
        )

        st.subheader(
            "Decision Distribution"
        )

        st.bar_chart(
            df["decision"].value_counts()
        )

    except:

        st.info(
            "Run video_timeline_v2.py first."
        )
        
        