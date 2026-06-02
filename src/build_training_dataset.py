import cv2
import numpy as np
import pandas as pd
import json
import os

IMAGE_FOLDER = "datasets/bdd100k/images"
LABEL_FILE = "datasets/bdd100k/labels/bdd100k_labels_images_train.json"

# Load labels
with open(LABEL_FILE, "r") as f:
    label_data = json.load(f)

# Create quick lookup
label_lookup = {}

for item in label_data:
    label_lookup[item["name"]] = item

results = []

images = os.listdir(IMAGE_FOLDER)

print(f"Found {len(images)} images")

for idx, image_name in enumerate(images):

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    image = cv2.resize(image, (640, 480))

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Brightness
    brightness = np.mean(gray)

    # Contrast
    contrast = gray.std()

    # Blur
    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # Edges
    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_count = np.sum(edges > 0)

    # Label lookup
    label = label_lookup.get(
        image_name,
        None
    )

    weather = "unknown"
    timeofday = "unknown"

    if label:

        attrs = label.get(
            "attributes",
            {}
        )

        weather = attrs.get(
            "weather",
            "unknown"
        )

        timeofday = attrs.get(
            "timeofday",
            "unknown"
        )

    results.append({
        "image_name": image_name,
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
        "edge_count": edge_count,
        "weather": weather,
        "timeofday": timeofday
    })

    if idx % 25 == 0:
        print(f"Processed {idx}")

df = pd.DataFrame(results)

os.makedirs("outputs", exist_ok=True)

df.to_csv(
    "outputs/bdd100k_features.csv",
    index=False
)

print("Dataset created successfully")
print(df.head())