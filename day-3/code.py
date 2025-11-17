import cv2
import numpy as np

# Read image
img_path = r"P:\Computer vision Experiments\30-Days challenge\CV-30-day-challenge\Images\deer.jpg"
img = cv2.imread(img_path, 0)

resizefactor = 0.7
h, w = img.shape[:2]
img = cv2.resize(img, (int(w*resizefactor), int(h*resizefactor)))

# Define structuring element
kernel = np.ones((5,5),np.uint8)

# Perform dilation
dilation = cv2.dilate(img, kernel, iterations=1)

# Perform erosion
erosion = cv2.erode(img, kernel, iterations=1)

# Opening
opened = cv2.dilate(erosion, kernel, iterations=1)

# Closing
closed = cv2.erode(dilation, kernel, iterations=1)

# Display results
cv2.imshow('Original',img)
cv2.imshow('Dilation',dilation)
cv2.imshow("Eroded", erosion)
cv2.imshow("Opened", opened)
cv2.imshow("Closed", closed)

cv2.waitKey(0)
cv2.destroyAllWindows()