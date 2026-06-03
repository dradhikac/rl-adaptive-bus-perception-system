from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image = cv2.imread("test_image.jpg")

results = model(image)

print("\nDetected Objects:")

for box in results[0].boxes:

    class_id = int(box.cls[0])

    class_name = model.names[class_id]

    confidence = float(box.conf[0])

    print(
        f"{class_name} : "
        f"{confidence:.2f}"
    )