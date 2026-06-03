from nuscenes.nuscenes import NuScenes
import cv2

# Dataset path
DATASET_PATH = "datasets/nuscenes"

# Load nuScenes
nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATASET_PATH,
    verbose=True
)

print("nuScenes loaded.")

# -----------------------------------
# Access first scene
# -----------------------------------

first_scene = nusc.scene[0]

print("\nScene Name:")
print(first_scene["name"])

print("\nDescription:")
print(first_scene["description"])

# -----------------------------------
# Access first sample
# -----------------------------------

first_sample_token = first_scene["first_sample_token"]

sample = nusc.get(
    "sample",
    first_sample_token
)

print("\nSample Token:")
print(sample["token"])

# -----------------------------------
# Available sensors
# -----------------------------------

print("\nAvailable Sensors:\n")

for sensor in sample["data"]:
    print(sensor)

# -----------------------------------
# Front camera
# -----------------------------------

cam_token = sample["data"]["CAM_FRONT"]

cam_data = nusc.get(
    "sample_data",
    cam_token
)

image_path = DATASET_PATH + "/" + cam_data["filename"]

print("\nImage Path:")
print(image_path)

# -----------------------------------
# Load image
# -----------------------------------

image = cv2.imread(image_path)

if image is None:
    print("Failed to load image.")
else:
    print("\nImage Shape:")
    print(image.shape)

    cv2.imshow("Front Camera", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()