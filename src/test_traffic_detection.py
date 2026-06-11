from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image = cv2.imread("test_image.jpg")

results = model(image)

print("\nDetected Objects:")

for box in results[0].boxes:

    cls_id = int(box.cls[0])

    name = model.names[cls_id]

    conf = float(box.conf[0])

    print(name, round(conf, 2))