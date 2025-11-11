import cv2
import numpy as np

image = cv2.imread(r"P:\Computer vision Experiments\30-Days challenge\CV-30-day-challenge\Images\image2.jpg")

(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# Rotation matrix: M = [[cosθ, sinθ, (1 - cosθ)*cx - sinθ*cy], ...]
M = cv2.getRotationMatrix2D(center, 30, 1)  # 45° rotation, scale = 1.0
rotated = cv2.warpAffine(image, M, (w, h))

cv2.imshow("Original", image)
cv2.imshow("Rotated", rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()