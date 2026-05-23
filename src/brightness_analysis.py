import cv2
import numpy as np
import os

image_folder = "datasets/bdd100k/images"

images = os.listdir(image_folder)

image_path = os.path.join(image_folder, images[0])

image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute average brightness
brightness = np.mean(gray)

print("Average Brightness:", brightness)