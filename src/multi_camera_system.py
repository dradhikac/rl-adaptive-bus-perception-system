from nuscenes.nuscenes import NuScenes
import cv2
import os
import matplotlib.pyplot as plt

# ============================================
# LOAD DATASET
# ============================================

DATASET_PATH = "datasets/nuscenes"

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATASET_PATH,
    verbose=False
)

print("Dataset loaded.")

# ============================================
# LOAD FIRST SAMPLE
# ============================================

scene = nusc.scene[0]

sample = nusc.get(
    "sample",
    scene["first_sample_token"]
)

# ============================================
# CAMERA LIST
# ============================================

camera_names = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT"
]

# ============================================
# LOAD ALL CAMERA IMAGES
# ============================================

camera_images = {}

for camera_name in camera_names:

    camera_token = sample["data"][camera_name]

    camera_data = nusc.get(
        "sample_data",
        camera_token
    )

    image_path = os.path.join(
        DATASET_PATH,
        camera_data["filename"]
    )

    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load {camera_name}")
        continue

    camera_images[camera_name] = image

    print(
        camera_name,
        image.shape
    )

# ============================================
# RESIZE IMAGES
# ============================================

for key in camera_images:

    camera_images[key] = cv2.resize(
        camera_images[key],
        (640, 360)
    )

# ============================================
# CREATE 360° SURROUND VIEW
# ============================================

top_row = cv2.hconcat([
    camera_images["CAM_FRONT_LEFT"],
    camera_images["CAM_FRONT"],
    camera_images["CAM_FRONT_RIGHT"]
])

bottom_row = cv2.hconcat([
    camera_images["CAM_BACK_LEFT"],
    camera_images["CAM_BACK"],
    camera_images["CAM_BACK_RIGHT"]
])

surround_view = cv2.vconcat([
    top_row,
    bottom_row
])

# ============================================
# DISPLAY USING MATPLOTLIB
# ============================================

surround_view_rgb = cv2.cvtColor(
    surround_view,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(18, 8))
plt.imshow(surround_view_rgb)
plt.title("360 Degree Camera System")
plt.axis("off")
plt.show()

print("\n360° view displayed successfully")