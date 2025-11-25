import cv2
import numpy as np
import math

image_path = r"P:\Computer vision Experiments\30-Days challenge\CV-30-day-challenge\Images\road.jpeg"

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

lines = cv2.Canny(gray, 100, 100)
hlines = cv2.HoughLines(lines, 1, np.pi/180, 100)

for line in hlines:
    rho, theta = line[0]

    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(image, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)


cv2.imshow("OG", gray)
cv2.imshow("Lines", lines)
cv2.imshow("Color", image)
cv2.waitKey(0)
cv2.destroyAllWindows()