import cv2
import numpy as np

# 1. Read the image and convert to gray
img = cv2.imread("../Images/elephant.jpg")  # put your image path here

resize_factor = 0.7
h, w = img.shape[:2]
img = cv2.resize(img, (int(resize_factor*w), int(resize_factor*h)))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Parameters for Shi-Tomasi
max_corners = 5000 # max number of corners to return
quality_level = 0.001 # minimum accepted quality (0–1, higher = fewer, stronger corners)
min_distance = 2 # minimum distance between corners (pixels)
block_size = 3 # size of neighborhood for covariance matrix

# 3. Detect corners using Shi-Tomasi (goodFeaturesToTrack)
corners = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=max_corners,
    qualityLevel=quality_level,
    minDistance=min_distance,
    blockSize=block_size,
    useHarrisDetector=False    # important: False = Shi-Tomasi, True = Harris
)

# corners is N x 1 x 2 array of (x, y)
corners = np.int8(corners)  # convert to int

# 4. Draw the detected corners on the image
vis = img.copy()
for c in corners:
    x, y = c.ravel()  # flatten
    cv2.circle(vis, (x, y), 4, (255, 0, 0), -1)  # red filled circle

# 5. Show result
cv2.imshow("Original", img)
cv2.imshow("Shi-Tomasi Corners", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()