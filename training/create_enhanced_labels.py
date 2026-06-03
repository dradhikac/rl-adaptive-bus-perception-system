import pandas as pd
import os

INPUT_FILE = "outputs/enhanced_bdd100k_features.csv"

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} samples")


def calculate_reliability(row):

    reliability = 1.0

    weather = str(row["weather"]).lower()
    timeofday = str(row["timeofday"]).lower()

    brightness = row["brightness"]
    contrast = row["contrast"]
    blur_score = row["blur_score"]

    object_count = row["object_count"]
    person_count = row["person_count"]
    vehicle_count = row["vehicle_count"]

    entropy = row["entropy"]

    # =========================
    # WEATHER
    # =========================

    weather_penalty = {
        "clear": 0.0,
        "partly cloudy": 0.10,
        "overcast": 0.20,
        "rainy": 0.35,
        "foggy": 0.45,
        "snowy": 0.50
    }

    reliability -= weather_penalty.get(
        weather,
        0.20
    )

    # =========================
    # TIME OF DAY
    # =========================

    if timeofday == "night":
        reliability -= 0.25

    elif timeofday == "dawn/dusk":
        reliability -= 0.10

    # =========================
    # IMAGE QUALITY
    # =========================

    if brightness < 60:
        reliability -= 0.10

    if contrast < 40:
        reliability -= 0.10

    if blur_score < 100:
        reliability -= 0.10

    # =========================
    # TRAFFIC COMPLEXITY
    # =========================

    if object_count > 15:
        reliability -= 0.05

    if person_count > 5:
        reliability -= 0.05

    if vehicle_count > 10:
        reliability -= 0.05

    # =========================
    # INFORMATION CONTENT
    # =========================

    if entropy < 5:
        reliability -= 0.05

    reliability = max(
        0.0,
        min(
            reliability,
            1.0
        )
    )

    return round(
        reliability,
        3
    )


df["reliability_score"] = df.apply(
    calculate_reliability,
    axis=1
)

OUTPUT_FILE = (
    "outputs/"
    "enhanced_training_dataset.csv"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nDistribution:")
print(
    df["reliability_score"]
    .describe()
)