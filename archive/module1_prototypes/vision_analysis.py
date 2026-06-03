import cv2
import numpy as np
import os
import pandas as pd

# =========================================================
# BDD100K Vision Reliability Analysis Engine
# =========================================================

# Dataset image folder
image_folder = "datasets/bdd100k/images"

# Get image list
images = os.listdir(image_folder)

# Check if images exist
if len(images) == 0:
    print("No images found in dataset folder.")
    exit()

# Load first image
image_path = os.path.join(image_folder, images[0])

image = cv2.imread(image_path)

# Check image loaded properly
if image is None:
    print("Failed to load image.")
    exit()

print("Image loaded successfully.")
print("Image Name:", images[0])

# =========================================================
# Resize Image
# =========================================================

image = cv2.resize(image, (640, 480))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# =========================================================
# BLUR DETECTION
# =========================================================

laplacian = cv2.Laplacian(gray, cv2.CV_64F)

blur_score = laplacian.var()

print("\nBlur Score:", blur_score)

# =========================================================
# EDGE DETECTION
# =========================================================

edges = cv2.Canny(gray, 100, 200)

# Count edge pixels
edge_count = np.sum(edges > 0)

print("Edge Pixel Count:", edge_count)

# =========================================================
# BRIGHTNESS ESTIMATION
# =========================================================

brightness = np.mean(gray)

print("Brightness:", brightness)

# =========================================================
# CONTRAST ANALYSIS
# =========================================================

contrast = gray.std()

print("Contrast:", contrast)

# =========================================================
# VISION RELIABILITY SCORING
# =========================================================

reliability = 1.0

# Blur impact
if blur_score < 100:
    reliability -= 0.3

# Darkness impact
if brightness < 60:
    reliability -= 0.2

# Low edge information
if edge_count < 5000:
    reliability -= 0.2

# Low contrast
if contrast < 40:
    reliability -= 0.2

# Clamp between 0 and 1
reliability = max(0.0, min(reliability, 1.0))

print("Vision Reliability Score:", reliability)

# =========================================================
# ENVIRONMENT DIFFICULTY SCORE
# =========================================================

difficulty_score = 1.0 - reliability

print("Environmental Difficulty Score:", difficulty_score)

# =========================================================
# SAVE RESULTS
# =========================================================

results = {
    "image_name": [images[0]],
    "blur_score": [blur_score],
    "brightness": [brightness],
    "edge_count": [edge_count],
    "contrast": [contrast],
    "reliability": [reliability],
    "difficulty_score": [difficulty_score]
}

df = pd.DataFrame(results)

# Create outputs folder if not exists
os.makedirs("outputs", exist_ok=True)

# Save CSV
df.to_csv("outputs/vision_analysis.csv", index=False)

print("\nVision analysis saved to outputs/vision_analysis.csv")

# =========================================================
# DISPLAY IMAGES
# =========================================================

cv2.imshow("Original Image", image)

cv2.imshow("Edge Detection", edges)

cv2.waitKey(0)

cv2.destroyAllWindows()