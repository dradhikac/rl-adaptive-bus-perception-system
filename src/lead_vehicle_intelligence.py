from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image = cv2.imread("test_image.jpg")

results = model(image)

height, width = image.shape[:2]

image_center = width // 2

best_vehicle = None

best_score = 999999

vehicle_classes = [

    "car",
    "truck",
    "bus",
    "motorcycle"
]

for box in results[0].boxes:

    class_id = int(box.cls[0])

    class_name = model.names[class_id]

    if class_name not in vehicle_classes:
        continue

    x1, y1, x2, y2 = box.xyxy[0]

    center_x = (x1 + x2) / 2

    area = (x2 - x1) * (y2 - y1)

    center_distance = abs(
        center_x - image_center
    )

    score = (center_distance * 2) - (area / 1000)

    if score < best_score:

        best_score = score

        best_vehicle = {

            "class": class_name,

            "area": area,

            "center_distance": center_distance
        }

print("\n")
print("========================")
print("LEAD VEHICLE")
print("========================")

print(
    "Vehicle:",
    best_vehicle["class"]
)

print(
    "Area:",
    int(best_vehicle["area"])
)

print(
    "Center Distance:",
    int(best_vehicle["center_distance"])
)