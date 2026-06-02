import pandas as pd
import os

# =========================================================
# LOAD FEATURES DATASET
# =========================================================

FEATURE_FILE = "outputs/bdd100k_features.csv"

df = pd.read_csv(FEATURE_FILE)

print(f"Loaded {len(df)} samples")

# =========================================================
# RELIABILITY CALCULATION FUNCTION
# =========================================================

def calculate_reliability(row):

    reliability = 1.0

    weather = str(row["weather"]).lower()
    timeofday = str(row["timeofday"]).lower()

    brightness = row["brightness"]
    contrast = row["contrast"]
    blur_score = row["blur_score"]
    edge_count = row["edge_count"]

    # =====================================================
    # WEATHER PENALTY
    # =====================================================

    weather_penalty = {
        "clear": 0.0,
        "partly cloudy": 0.1,
        "overcast": 0.2,
        "rainy": 0.4,
        "foggy": 0.5,
        "snowy": 0.5
    }

    reliability -= weather_penalty.get(weather, 0.2)

    # =====================================================
    # TIME PENALTY
    # =====================================================

    if timeofday == "night":
        reliability -= 0.3

    elif timeofday == "dawn/dusk":
        reliability -= 0.15

    # =====================================================
    # IMAGE QUALITY PENALTIES
    # =====================================================

    if brightness < 60:
        reliability -= 0.10

    if contrast < 40:
        reliability -= 0.10

    if blur_score < 100:
        reliability -= 0.10

    if edge_count < 5000:
        reliability -= 0.10

    # =====================================================
    # CLAMP BETWEEN 0 AND 1
    # =====================================================

    reliability = max(0.0, min(reliability, 1.0))

    return round(reliability, 3)

# =========================================================
# GENERATE LABELS
# =========================================================

df["reliability_score"] = df.apply(
    calculate_reliability,
    axis=1
)

# =========================================================
# SAVE TRAINING DATASET
# =========================================================

os.makedirs("outputs", exist_ok=True)

OUTPUT_FILE = "outputs/bdd100k_training_dataset.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nTraining dataset created successfully")
print(f"Saved to: {OUTPUT_FILE}")

print("\nReliability Distribution:")

print(
    df["reliability_score"]
    .describe()
)

print("\nFirst 5 Samples:")

print(
    df[
        [
            "image_name",
            "weather",
            "timeofday",
            "reliability_score"
        ]
    ].head()
)