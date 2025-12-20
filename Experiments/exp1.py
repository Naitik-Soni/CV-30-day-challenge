import cv2
import numpy as np

image_path = r"../Images/garden.jpeg"

image = cv2.imread(image_path)

rf = 0.3
image = cv2.resize(image, None, None, fx=rf, fy=rf, interpolation=cv2.INTER_AREA)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(hsv)

lower = np.array([85, 0, 0])
upper = np.array([135, 255, 255])

mask = cv2.inRange(hsv, lower, upper)

new_image = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow("Garden", image)
cv2.imshow("HSV", hsv)
cv2.imshow("H", h)
cv2.imshow("S", s)
cv2.imshow("V", v)
cv2.imshow("New image", new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()