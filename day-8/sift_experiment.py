import cv2

img = cv2.imread(r"..\Images\cube.jpg", 0)

resize_Factor = 0.5
h, w = img.shape
img3 = cv2.resize(img, (int(w * resize_Factor), int(h * resize_Factor)))

resize_Factor = 0.7
h, w = img.shape
img1 = cv2.resize(img, (int(w * resize_Factor), int(h * resize_Factor)))

resize_Factor = 0.9
h, w = img.shape
img2 = cv2.resize(img, (int(w * resize_Factor), int(h * resize_Factor)))

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(img3, None)
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

out = cv2.drawKeypoints(img3, keypoints, None)
out1 = cv2.drawKeypoints(img1, keypoints1, None)
out2 = cv2.drawKeypoints(img2, keypoints2, None)

cv2.imshow("SIFT keypoints", out)
cv2.imshow("SIFT keypoints 0.7", out1)
cv2.imshow("SIFT keypoints 0.9", out2)
cv2.waitKey(0)
cv2.destroyAllWindows()