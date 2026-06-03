from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

IMAGE_PATH = "test_image.jpg"

image = cv2.imread(IMAGE_PATH)

results = model(image)

closest_vehicle = None
largest_area = 0

vehicle_classes = [
    "car",
    "truck",
    "bus",
    "motorcycle"
]

for box in results[0].boxes:

    cls_id = int(box.cls[0])

    cls_name = model.names[cls_id]

    if cls_name not in vehicle_classes:
        continue

    x1, y1, x2, y2 = box.xyxy[0]

    width = x2 - x1
    height = y2 - y1

    area = width * height

    if area > largest_area:

        largest_area = area

        closest_vehicle = cls_name

print("\n====================")
print("VEHICLE DISTANCE")
print("====================")

print(
    "Closest Vehicle:",
    closest_vehicle
)

print(
    "Bounding Box Area:",
    int(largest_area)
)

if largest_area > 150000:

    distance_state = "VERY CLOSE"

elif largest_area > 50000:

    distance_state = "MEDIUM"

else:

    distance_state = "FAR"

print(
    "Distance Estimate:",
    distance_state
)

print("====================")