import cv2
import numpy as np

image_path = r"P:\Computer vision Experiments\30-Days challenge\CV-30-day-challenge\Images\bed.jpg"

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

lines = cv2.Canny(gray, 100, 100)

circles = cv2.HoughCircles(lines, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10, param1=50, param2=30, minRadius=5, maxRadius=100)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow("GRAY", gray)
cv2.imshow("CIRCLES", lines)
cv2.imshow("HOUGH", image)

cv2.waitKey(0)
cv2.destroyAllWindows()