import cv2
import numpy as np

# Read image
img_path = r"P:\Computer vision Experiments\30-Days challenge\CV-30-day-challenge\Images\bed.jpg"
img = cv2.imread(img_path, 0)

resizefactor = 0.5
w, h = img.shape[:2]
img = cv2.resize(img, (int(w*resizefactor), int(h*resizefactor)))

# Define structuring element
kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((11,11),np.uint8)
kernel3 = np.ones((25,25),np.uint8)

# Perform dilation
dilation1 = cv2.dilate(img, kernel, iterations = 1)
dilation5 = cv2.dilate(img, kernel2, iterations = 1)
dilation9 = cv2.dilate(img, kernel3, iterations = 1)

# Display results
cv2.imshow('Original',img)
cv2.imshow('Dilation1',dilation1)
cv2.imshow('Dilation5',dilation5)
cv2.imshow('Dilation9',dilation9)
cv2.waitKey(0)
cv2.destroyAllWindows()