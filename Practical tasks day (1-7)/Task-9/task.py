import cv2
import numpy as np

image_path = "fire-2.jpg"
image = cv2.imread(image_path)

h, w = image.shape[:2]
resize_factor = 0.4*2.5
new_dimensions = (int(w * resize_factor), int(h * resize_factor))
resized_image = cv2.resize(image, new_dimensions)

gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

min_fire = np.array([0, 100, 150])
max_fire = np.array([33, 255, 255])
fire_mask = cv2.inRange(hsv_image, min_fire, max_fire)
fire_mask = np.bitwise_and(gray_image, fire_mask)

cv2.imshow("Gray Image", gray_image)
cv2.imshow("Resized Image", resized_image)
cv2.imshow("HSV Image", hsv_image)
cv2.imshow("Fire Mask", fire_mask)
cv2.waitKey(0)