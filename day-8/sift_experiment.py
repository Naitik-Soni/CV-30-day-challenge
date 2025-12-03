import cv2

img = cv2.imread(r"..\Images\phone.jpg", 0)

resize_Factor = 0.3
h, w = img.shape
img = cv2.resize(img, (int(w * resize_Factor), int(h * resize_Factor)))

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(img, None)

out = cv2.drawKeypoints(img, keypoints, None)

cv2.imshow("SIFT keypoints", out)
cv2.waitKey(0)
cv2.destroyAllWindows()