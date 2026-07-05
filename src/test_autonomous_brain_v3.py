import cv2
from pprint import pprint

from autonomous_brain_v3 import analyze_scene

# Replace with your actual image paths

front = cv2.imread(
    "data/front.jpg"
)

front_left = cv2.imread(
    "data/front_left.jpg"
)

front_right = cv2.imread(
    "data/front_right.jpg"
)

rear_left = cv2.imread(
    "data/rear_left.jpg"
)

rear_right = cv2.imread(
    "data/rear_right.jpg"
)

result = analyze_scene(
    front,
    front_left,
    front_right,
    rear_left,
    rear_right
)

print("\n===== AUTONOMOUS BRAIN OUTPUT =====\n")

pprint(result)