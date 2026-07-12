import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(PROJECT_ROOT)

from src.autonomous_brain_v3 import analyze_scene


# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="Autonomous Bus AI Dashboard V3",
    page_icon="🚍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================
# GLOBAL STYLE
# ====================================

CUSTOM_CSS = """
<style>

:root {
    --navy: #0B1F3A;
    --navy-light: #12294D;
    --teal: #14B8A6;
    --orange: #F97316;
    --grey-bg: #F4F6F9;
    --card-border: #E4E8F0;
}

/* ---------- Base ---------- */
.stApp {
    background-color: var(--grey-bg);
}

#MainMenu, footer {visibility: hidden;}

/* ---------- Hero header ---------- */
.hero-banner {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 60%, #163A63 100%);
    padding: 2.2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 24px rgba(11, 31, 58, 0.25);
}

.hero-title {
    color: #FFFFFF;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: 0.3px;
}

.hero-subtitle {
    color: #B9C6DC;
    font-size: 1rem;
    margin-top: 0.35rem;
    font-weight: 400;
}

.hero-badges {
    margin-top: 1rem;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.hero-badge {
    background: rgba(20, 184, 166, 0.15);
    color: var(--teal);
    border: 1px solid rgba(20, 184, 166, 0.4);
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* ---------- Section headers ---------- */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.4rem 0 0.8rem 0;
}

.section-header .bar {
    width: 5px;
    height: 22px;
    background: var(--orange);
    border-radius: 3px;
}

.section-header h3 {
    margin: 0;
    color: var(--navy);
    font-weight: 700;
}

/* ---------- Metric cards ---------- */
.metric-card {
    background: #FFFFFF;
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    text-align: left;
    box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
    height: 100%;
}

.metric-label {
    font-size: 0.78rem;
    color: #6B7488;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--navy);
    line-height: 1.2;
}

.metric-icon {
    font-size: 1.1rem;
    margin-right: 0.35rem;
}

.accent-teal { border-left: 4px solid var(--teal); }
.accent-orange { border-left: 4px solid var(--orange); }
.accent-navy { border-left: 4px solid var(--navy); }
.accent-red { border-left: 4px solid #E11D48; }

/* ---------- Camera panel ---------- */
.cam-caption {
    background: var(--navy);
    color: white;
    display: inline-block;
    padding: 0.15rem 0.7rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.4rem;
}

/* ---------- Status pills ---------- */
.pill {
    display: inline-block;
    padding: 0.45rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
}

.pill-red { background: #FDE8E8; color: #C81E1E; }
.pill-yellow { background: #FEF3C7; color: #92640B; }
.pill-green { background: #DCFCE7; color: #15803D; }
.pill-grey { background: #E5E7EB; color: #374151; }

/* ---------- Uploader card ---------- */
.upload-card {
    background: #FFFFFF;
    border: 1px dashed var(--card-border);
    border-radius: 14px;
    padding: 0.9rem 1rem 0.3rem 1rem;
    margin-bottom: 0.8rem;
}

/* ---------- Reason list ---------- */
.reason-item {
    background: #FFF7ED;
    border-left: 3px solid var(--orange);
    padding: 0.5rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    color: #7C2D12;
    font-size: 0.9rem;
}

/* ---------- Range bars ---------- */
.range-track {
    position: relative;
    height: 7px;
    background: #EDEFF4;
    border-radius: 4px;
    margin-top: 0.7rem;
}

.range-fill {
    height: 7px;
    border-radius: 4px;
}

.range-marker {
    position: absolute;
    top: -3px;
    width: 2px;
    height: 13px;
    background: #1F2937;
    opacity: 0.55;
}

.range-scale {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: #9AA3B2;
    margin-top: 0.25rem;
}

.range-note {
    font-size: 0.7rem;
    color: #6B7488;
    margin-top: 0.15rem;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: #FFFFFF;
    padding: 0.35rem;
    border-radius: 12px;
    border: 1px solid var(--card-border);
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 8px;
    padding: 0 16px;
    font-weight: 600;
    color: #5B6474;
}

.stTabs [aria-selected="true"] {
    background-color: var(--navy) !important;
    color: white !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--navy);
}

section[data-testid="stSidebar"] * {
    color: #EAF0FB !important;
}

section[data-testid="stSidebar"] .stAlert {
    background-color: rgba(20, 184, 166, 0.12) !important;
    border: 1px solid rgba(20, 184, 166, 0.35) !important;
    border-radius: 10px !important;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ====================================
# HELPERS
# ====================================

def metric_card(label, value, icon="", accent="navy"):
    st.markdown(
        f"""
        <div class="metric-card accent-{accent}">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def metric_card_ranged(label, value_display, icon, accent, raw_value, min_val, max_val, threshold, unit=""):
    """Metric card with a small range bar showing where raw_value falls
    between min_val and max_val, plus a marker at the acceptable threshold."""

    span = max_val - min_val
    fill_pct = max(0, min(100, ((raw_value - min_val) / span) * 100))
    marker_pct = max(0, min(100, ((threshold - min_val) / span) * 100))

    good = raw_value >= threshold
    bar_color = "var(--teal)" if good else "var(--orange)"
    status_text = "Good" if good else "Below threshold"

    st.markdown(
        f"""
        <div class="metric-card accent-{accent}">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value_display}{unit}</div>
            <div class="range-track">
                <div class="range-fill" style="width:{fill_pct}%; background:{bar_color};"></div>
                <div class="range-marker" style="left:{marker_pct}%;"></div>
            </div>
            <div class="range-scale">
                <span>{min_val}</span>
                <span>{max_val}</span>
            </div>
            <div class="range-note">Min acceptable: {threshold}{unit} &nbsp;•&nbsp; {status_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_header(title, icon=""):
    st.markdown(
        f"""
        <div class="section-header">
            <div class="bar"></div>
            <h3>{icon} {title}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )


def status_pill(text, kind="grey"):
    st.markdown(
        f'<span class="pill pill-{kind}">{text}</span>',
        unsafe_allow_html=True
    )


# ====================================
# HERO HEADER
# ====================================

st.markdown(
    """
    <div class="hero-banner">
        <p class="hero-title">🚍 Autonomous Bus AI Dashboard</p>
        <p class="hero-subtitle">360° Perception &nbsp;•&nbsp; Blind Spot Intelligence &nbsp;•&nbsp; Reinforcement-Learning Decision Making</p>
        <div class="hero-badges">
            <span class="hero-badge">V3 Engine</span>
            <span class="hero-badge">Multi-Camera Fusion</span>
            <span class="hero-badge">Real-Time Risk Scoring</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ====================================
# SIDEBAR
# ====================================

st.sidebar.markdown("## 🧭 System Status")
st.sidebar.markdown("---")

sidebar_items = [
    ("Reliability Intelligence", "📷"),
    ("Sensor Fusion", "📡"),
    ("Traffic Intelligence", "🚦"),
    ("Lead Vehicle Intelligence", "🚗"),
    ("Blind Spot Intelligence", "👀"),
    ("RL Decision Engine", "🤖"),
    ("Decision Fusion", "🧠"),
    ("Autonomous Brain V3", "🚍"),
]

for label, icon in sidebar_items:
    st.sidebar.success(f"{icon}  {label} — Online")

st.sidebar.markdown("---")
st.sidebar.caption("All subsystems operational. Awaiting camera input.")

# ====================================
# IMAGE LOADER
# ====================================

def load_image(uploaded_file):

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image = np.array(image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    return image

# ====================================
# CAMERA INPUTS
# ====================================

section_header("360° Camera Inputs", "📸")

st.markdown('<div class="upload-card">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:

    front_image = st.file_uploader(
        "⬆️ Front Camera",
        type=["jpg", "jpeg", "png"]
    )

    front_left_image = st.file_uploader(
        "↖️ Front Left Camera",
        type=["jpg", "jpeg", "png"]
    )

    rear_left_image = st.file_uploader(
        "↙️ Rear Left Camera",
        type=["jpg", "jpeg", "png"]
    )

with col2:

    front_right_image = st.file_uploader(
        "↗️ Front Right Camera",
        type=["jpg", "jpeg", "png"]
    )

    rear_right_image = st.file_uploader(
        "↘️ Rear Right Camera",
        type=["jpg", "jpeg", "png"]
    )

st.markdown('</div>', unsafe_allow_html=True)

# ====================================
# ANALYSIS
# ====================================

if all([

    front_image,
    front_left_image,
    front_right_image,
    rear_left_image,
    rear_right_image

]):

    front_np = load_image(front_image)

    front_left_np = load_image(
        front_left_image
    )

    front_right_np = load_image(
        front_right_image
    )

    rear_left_np = load_image(
        rear_left_image
    )

    rear_right_np = load_image(
        rear_right_image
    )

    result = analyze_scene(

        front_np,

        front_left_np,

        front_right_np,

        rear_left_np,

        rear_right_np
    )

    # ====================================
    # OVERVIEW
    # ====================================

    section_header("Autonomous Driving Overview", "📊")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        metric_card("Signal", result["traffic_signal"], "🚦", "orange")

    with c2:
        metric_card("Objects", result["object_count"], "🔷", "navy")

    with c3:
        risk_level_val = str(result["risk_level"]).upper()
        risk_accent = "teal" if risk_level_val == "LOW" else ("orange" if risk_level_val in ("MEDIUM", "MED") else "red")
        metric_card("Risk", result["risk_level"], "⚠️", risk_accent)

    with c4:
        metric_card("RL Action", result["rl_action"], "🤖", "teal")

    with c5:
        metric_card("Decision", result["decision"], "🧠", "navy")

    with c6:
        metric_card("Speed", f"{result['recommended_speed']} km/h", "⚡", "teal")

    st.markdown("<br>", unsafe_allow_html=True)

    # ====================================
    # TABS
    # ====================================

    tabs = st.tabs([

        "🚍 Perception",

        "🚦 Traffic",

        "🚗 Lead Vehicle",

        "📷 Reliability",

        "⚠️ Risk",

        "👀 Blind Spot",

        "🤖 RL",

        "🧠 Decision Fusion",

        "🚍 Autonomous Brain"
    ])

    # ====================================
    # PERCEPTION
    # ====================================

    with tabs[0]:

        section_header("360° Multi-Camera Perception", "🚍")

        # ---------- FRONT ----------

        st.markdown("#### 🚍 Front Camera")

        st.image(
            result["front_annotated"],
            use_container_width=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card("Objects", result["object_count"], "🔷", "navy")

        with c2:
            metric_card("Vehicles", result["vehicle_count"], "🚗", "teal")

        with c3:
            metric_card("Pedestrians", result["pedestrian_count"], "🚶", "orange")

        st.divider()

        # ---------- LEFT SIDE ----------

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### ↖ Front Left Camera")

            st.image(
                result["front_left_annotated"],
                use_container_width=True
            )

            st.markdown(
                f'<span class="cam-caption">Objects: {result["front_left_objects"]}</span>',
                unsafe_allow_html=True
            )

        with col2:

            st.markdown("#### ↙ Rear Left Camera")

            st.image(
                result["rear_left_annotated"],
                use_container_width=True
            )

            st.markdown(
                f'<span class="cam-caption">Objects: {result["rear_left_objects"]}</span>',
                unsafe_allow_html=True
            )

        st.divider()

        # ---------- RIGHT SIDE ----------

        col3, col4 = st.columns(2)

        with col3:

            st.markdown("#### ↗ Front Right Camera")

            st.image(
                result["front_right_annotated"],
                use_container_width=True
            )

            st.markdown(
                f'<span class="cam-caption">Objects: {result["front_right_objects"]}</span>',
                unsafe_allow_html=True
            )

        with col4:

            st.markdown("#### ↘ Rear Right Camera")

            st.image(
                result["rear_right_annotated"],
                use_container_width=True
            )

            st.markdown(
                f'<span class="cam-caption">Objects: {result["rear_right_objects"]}</span>',
                unsafe_allow_html=True
            )

        st.divider()

        # ---------- BLIND SPOT SUMMARY ----------

        section_header("Blind Spot Occupancy Summary", "👀")

        b1, b2, b3 = st.columns(3)

        with b1:
            metric_card("Left Lane", result["left_lane_status"], "⬅️", "navy")

        with b2:
            metric_card("Right Lane", result["right_lane_status"], "➡️", "navy")

        with b3:
            metric_card("Collision Risk", result["collision_risk"], "⚠️", "red")

    # ====================================
    # TRAFFIC
    # ====================================

    with tabs[1]:

        section_header("Traffic Signal Status", "🚦")

        signal = result["traffic_signal"]

        if signal == "RED":
            status_pill("🔴  RED SIGNAL — Stop", "red")

        elif signal == "YELLOW":
            status_pill("🟡  YELLOW SIGNAL — Caution", "yellow")

        elif signal == "GREEN":
            status_pill("🟢  GREEN SIGNAL — Go", "green")

        else:
            status_pill("⚪  UNKNOWN", "grey")

    # ====================================
    # LEAD VEHICLE
    # ====================================

    with tabs[2]:

        section_header("Lead Vehicle Intelligence", "🚗")

        l1, l2, l3 = st.columns(3)

        with l1:
            metric_card("Lead Vehicle", result["lead_vehicle"], "🚗", "navy")

        with l2:
            metric_card("Vehicle Area", result["lead_vehicle_area"], "📐", "teal")

        with l3:
            metric_card("Distance", result["vehicle_distance"], "📏", "orange")

    # ====================================
    # RELIABILITY
    # ====================================

    with tabs[3]:

        section_header("Camera Reliability Diagnostics", "📷")

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            metric_card_ranged(
                "Brightness", result["brightness"], "☀️", "orange",
                raw_value=result["brightness"], min_val=0, max_val=255, threshold=60
            )

        with r2:
            metric_card_ranged(
                "Contrast", result["contrast"], "◐", "navy",
                raw_value=result["contrast"], min_val=0, max_val=120, threshold=40
            )

        with r3:
            metric_card_ranged(
                "Blur Score", result["blur_score"], "🌫️", "teal",
                raw_value=result["blur_score"], min_val=0, max_val=500, threshold=100
            )

        with r4:
            metric_card_ranged(
                "Reliability", result["camera_reliability"], "✅", "teal",
                raw_value=result["camera_reliability"], min_val=0, max_val=1, threshold=0.6
            )

    
    # ====================================
    # RISK
    # ====================================

    with tabs[4]:

        section_header("Risk Assessment", "⚠️")

        rk1, rk2 = st.columns(2)

        risk_level_val = str(result["risk_level"]).upper()
        risk_accent = "teal" if risk_level_val == "LOW" else ("orange" if risk_level_val in ("MEDIUM", "MED") else "red")

        with rk1:
            metric_card("Risk Level", result["risk_level"], "🎯", risk_accent)

        with rk2:
            metric_card("Risk Score", result["risk_score"], "⚠️", "navy")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Contributing Factors:**")

        for reason in result["reasons"]:
            st.markdown(
                f'<div class="reason-item">• {reason}</div>',
                unsafe_allow_html=True
            )

    # ====================================
    # BLIND SPOT
    # ====================================

    with tabs[5]:

        section_header("Blind Spot Intelligence", "👀")

        c1, c2 = st.columns(2)

        with c1:
            metric_card("Left Lane", result["left_lane_status"], "⬅️", "navy")

        with c2:
            metric_card("Right Lane", result["right_lane_status"], "➡️", "navy")

        st.markdown("<br>", unsafe_allow_html=True)

        metric_card("Collision Risk", result["collision_risk"], "⚠️", "red")

        st.markdown("<br>", unsafe_allow_html=True)

        st.success(
            result["lane_recommendation"]
        )

    # ====================================
    # RL
    # ====================================

    with tabs[6]:

        section_header("Reinforcement Learning Decision", "🤖")

        metric_card("RL Action", result["rl_action"], "🤖", "teal")

    # ====================================
    # DECISION FUSION
    # ====================================

    with tabs[7]:

        section_header("Decision Fusion", "🧠")

        metric_card("Lane Action", result["lane_action"], "🧭", "navy")

        st.markdown("<br>", unsafe_allow_html=True)

        st.info(
            result["fusion_reason"]
        )

    # ====================================
    # AUTONOMOUS BRAIN
    # ====================================

    with tabs[8]:

        section_header("Autonomous Brain — Final Output", "🚍")

        f1, f2 = st.columns(2)

        with f1:
            metric_card("Final Decision", result["decision"], "🧠", "navy")

        with f2:
            metric_card("Recommended Speed", f"{result['recommended_speed']} km/h", "⚡", "teal")

        f3, f4, f5 = st.columns(3)

        with f3:
            metric_card("Risk Level", result["risk_level"], "⚠️", "red")

        with f4:
            metric_card("RL Action", result["rl_action"], "🤖", "teal")

        with f5:
            metric_card("Lane Action", result["lane_action"], "🧭", "orange")

else:

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("📸 Upload all five camera feeds above (Front, Front-Left, Front-Right, Rear-Left, Rear-Right) to run the full perception and decision pipeline.")
