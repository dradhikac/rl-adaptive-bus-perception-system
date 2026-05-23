import cv2
import os

# Path to image folder
image_folder = "datasets/bdd100k/images"

# List all image files
images = os.listdir(image_folder)

print("Total Images:", len(images))

# Load first image
first_image_path = os.path.join(image_folder, images[0])

image = cv2.imread(first_image_path)

# Print image details
print("Image Shape:", image.shape)

# Show image
cv2.imshow("BDD100K Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()