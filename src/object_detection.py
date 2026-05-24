from ultralytics import YOLO
import cv2
import os

# Load model
model = YOLO("yolov8n.pt")

# Load image
image_folder = "datasets/bdd100k/images"

images = os.listdir(image_folder)

image_path = os.path.join(image_folder, images[0])

image = cv2.imread(image_path)

# Run detection
results = model(image)

# Annotated image
annotated_frame = results[0].plot()

# Save output image

cv2.imwrite(
    "outputs/detected_image.jpg",
    annotated_frame
)

print("Annotated image saved.")

# Extract detection information

for box in results[0].boxes:

    class_id = int(box.cls[0])

    confidence = float(box.conf[0])

    class_name = model.names[class_id]

    print("------------------------")
    print("Object:", class_name)
    print("Confidence:", confidence)
    
    object_count = {}

for box in results[0].boxes:

    class_id = int(box.cls[0])

    class_name = model.names[class_id]

    if class_name not in object_count:
        object_count[class_name] = 0

    object_count[class_name] += 1

print("\nObject Counts:")
print(object_count)

import pandas as pd

detections = []

for box in results[0].boxes:

    class_id = int(box.cls[0])

    confidence = float(box.conf[0])

    class_name = model.names[class_id]

    detections.append({
        "object": class_name,
        "confidence": confidence
    })

df = pd.DataFrame(detections)

df.to_csv(
    "outputs/object_detection_results.csv",
    index=False
)

print("Detection CSV saved.")


# Show image
cv2.imshow("YOLO Detection", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()