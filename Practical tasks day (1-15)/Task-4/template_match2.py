# Python program to illustrate
# template matching
import cv2
import numpy as np

def resize(img, resize_factor = 0.5):
    return cv2.resize(img, None, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

# Read the main image
img_rgb = cv2.imread(r'./car.jpeg')
img_rgb = resize(img_rgb)

# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Read the template
template = cv2.imread(r'./logo.webp', 0)
template = resize(template, resize_factor=0.2)

# Store width and height of template in w and h
w, h = template.shape[::-1]

# Perform match operations.
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

# Specify a threshold
threshold = 0.6

# Store the coordinates of matched area in a numpy array
loc = np.where(res >= threshold)

# Draw a rectangle around the matched region.
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 240, 255), 3)

# Show the final image with the matched area.
cv2.imshow('Detected', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()