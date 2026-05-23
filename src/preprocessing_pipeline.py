import cv2
import os
import numpy as np

# Dataset image folder
image_folder = "datasets/bdd100k/images"

# Read all image names
images = os.listdir(image_folder)

# Select first image
image_path = os.path.join(image_folder, images[0])

# Load image
image = cv2.imread(image_path)

# Original shape
print("Original Shape:", image.shape)

# Resize image
resized_image = cv2.resize(image, (640, 480))

print("Resized Shape:", resized_image.shape)

# Normalize image
normalized_image = resized_image / 255.0

print("Pixel Range After Normalization:")
print("Min:", normalized_image.min())
print("Max:", normalized_image.max())

# Display original image
cv2.imshow("Original Image", image)

# Display resized image
cv2.imshow("Resized Image", resized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()