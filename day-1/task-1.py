# Load and display an image using OpenCV.
# Convert it to: Grayscale, HSV color space, Resize and rotate it
# Save all versions

import cv2

image_path = r"P:\Computer vision Experiments\30-Days challenge\CV-30-day-challenge\Images\garden.jpg"

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h, w = image.shape[:2]

resize_factor = 0.4
resized = cv2.resize(image, (int(resize_factor*w), int(resize_factor*h)))

rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

cv2.imwrite("Original.png", image)
cv2.imwrite("Gray.png", gray)
cv2.imwrite("HSV.png", hsv)
cv2.imwrite("Resized.png", resized)
cv2.imwrite("Rotated.png", rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()