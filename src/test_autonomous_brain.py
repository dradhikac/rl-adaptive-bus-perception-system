from autonomous_brain import analyze_scene
import cv2

image = cv2.imread(
    "test_image.jpg"
)

result = analyze_scene(
    image
)

for key, value in result.items():

    if key != "annotated_image":

        print(
            f"{key}: {value}"
        )