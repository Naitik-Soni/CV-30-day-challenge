import cv2
import numpy as np

img = cv2.imread(r"../Images/autumn.jpg")

tx, ty = 100, 50  # shift right 100, down 50
M = np.float32([[1, 0, tx],
                [0, 1, ty]])

translated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

cv2.imshow("Translated", translated)
cv2.waitKey(0)
