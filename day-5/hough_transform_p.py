import cv2
import numpy as np

image_path = r"P:\Computer vision Experiments\30-Days challenge\CV-30-day-challenge\Images\bed.jpg"

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

lines = cv2.Canny(gray, 100, 100)

hlines = cv2.HoughLinesP(lines, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

for line in hlines:
    x1, y1, x2, y2 = line[0]

    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_8)

cv2.imshow("Lines", lines)
cv2.imshow("Gray", gray)
cv2.imshow("Colored", image)

cv2.waitKey(0)
cv2.destroyAllWindows()