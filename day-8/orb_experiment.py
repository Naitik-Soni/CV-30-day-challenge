import cv2
import numpy as np

img = cv2.imread(r"..\Images\balcony garden.jpg", 0)

resize_factor = 0.4
h, w = img.shape[:2]
img = cv2.resize(img, (int(w * resize_factor), int(h * resize_factor)))

# Create ORB detector with default settings
orb = cv2.ORB_create()

# Detect keypoints and descriptors
keypoints, descriptors = orb.detectAndCompute(img, None)

print(f"Number of keypoints detected: {len(keypoints)}")
print(f"Descriptor shape: {descriptors.shape}")

print(keypoints[0].pt)  # Print coordinates of the first keypoint
print(descriptors[0])    # Print descriptor of the first keypoint

# Draw keypoints
out = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0))

cv2.imshow("ORB Keypoints", out)
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
