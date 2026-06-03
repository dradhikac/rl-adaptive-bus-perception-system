from ultralytics import YOLO
import cv2

# ====================================
# LOAD MODEL
# ====================================

model = YOLO("yolov8n.pt")

# ====================================
# LOAD IMAGE
# ====================================

image = cv2.imread("test_image.jpg")

results = model(image)

height, width = image.shape[:2]

image_center = width // 2

# ====================================
# FIND LEAD VEHICLE
# ====================================

best_vehicle = None

best_score = -999999

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

    area = (x2 - x1) * (y2 - y1)

    center_x = (x1 + x2) / 2

    center_distance = abs(
        center_x - image_center
    )

    score = (

        area / 1000

    ) - (

        center_distance
    )

    if score > best_score:

        best_score = score

        best_vehicle = {

            "class": class_name,

            "area": area
        }

# ====================================
# GAP ESTIMATION
# ====================================

area = best_vehicle["area"]

if area > 150000:

    gap_meters = 1.0

elif area > 120000:

    gap_meters = 2.0

elif area > 80000:

    gap_meters = 4.0

elif area > 40000:

    gap_meters = 8.0

else:

    gap_meters = 15.0

# ====================================
# SAFETY STATUS
# ====================================

if gap_meters <= 2:

    status = "UNSAFE"

elif gap_meters <= 5:

    status = "CAUTION"

else:

    status = "SAFE"

# ====================================
# REPORT
# ====================================

print("\n")
print("=" * 35)
print(" FOLLOWING DISTANCE REPORT ")
print("=" * 35)

print(
    "Lead Vehicle:",
    best_vehicle["class"]
)

print(
    "Estimated Gap:",
    gap_meters,
    "m"
)

print(
    "Status:",
    status
)

print("=" * 35)