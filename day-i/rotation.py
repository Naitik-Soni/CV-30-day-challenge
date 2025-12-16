import cv2
import numpy as np

img = cv2.imread(r"../Images/autumn.jpg")

(h, w) = img.shape[:2]
center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
rotated = cv2.warpAffine(img, M, (w, h))

cv2.imshow("Rotated", rotated)
cv2.waitKey(0)
