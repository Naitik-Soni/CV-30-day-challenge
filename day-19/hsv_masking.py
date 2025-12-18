import cv2
import numpy as np

image = cv2.imread(r"../Images/colorized.jpeg")

rf = 0.3
image = cv2.resize(image, None, None, fx=rf, fy=rf, interpolation=cv2.INTER_AREA)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(hsv)

lower = np.array([35, 0, 0])
upper = np.array([85, 255, 255])

lower1 = np.array([160, 0, 0])
upper1 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower, upper)
mask2 = cv2.inRange(hsv, lower1, upper1)

mask = mask1

kernel = np.ones((3,3), np.uint8)
maskm = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
maskm = cv2.morphologyEx(maskm, cv2.MORPH_CLOSE, kernel)

maskm = cv2.morphologyEx(maskm, cv2.MORPH_OPEN, kernel)
maskm = cv2.morphologyEx(maskm, cv2.MORPH_CLOSE, kernel)

result = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow("BGR", image)
cv2.imshow("Maskm", maskm)
cv2.imshow("Result", result)
cv2.imshow("Mask 1", mask1)
# cv2.imshow("Mask 2", mask2)
cv2.imshow("Mask", mask)
cv2.imshow("H", h)
cv2.imshow("S", s)
cv2.imshow("V", v)

cv2.waitKey(0)