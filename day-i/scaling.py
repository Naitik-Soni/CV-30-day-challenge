import cv2
import numpy as np

img = cv2.imread(r"../Images/full moon.jpg")

resized = cv2.resize(img, None, fx=2.35, fy=2.35, interpolation=cv2.INTER_LINEAR)

cv2.imshow("Resized og", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()