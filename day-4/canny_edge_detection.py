import cv2
import numpy as np
from non_max_suppression import non_max_suppression_interpolated

image_path = r"P:\Computer vision Experiments\30-Days challenge\CV-30-day-challenge\Images\deer.jpg"

img = cv2.imread(image_path)

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gaussian blur
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Gradients
gx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
gy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)

# Magnitude + Direction
magnitude = np.hypot(gx, gy)
direction = np.arctan2(gy, gx)

# Non max supression
nms_simple = non_max_suppression_interpolated(magnitude, direction)

cv2.imshow("NMS", nms_simple)
cv2.waitKey(0)
cv2.destroyAllWindows()