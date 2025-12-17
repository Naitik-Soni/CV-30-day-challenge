import cv2

image_path = r"C:\Users\baps\Documents\Naitik Soni\ComputerVision\CV-30-day-challenge\Practical tasks day (1-7)\Task-5\coins-1.jpg"
img = cv2.imread(image_path)


params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.7

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(img)

out = cv2.drawKeypoints(img, keypoints, None,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow("Original image", img)
cv2.imshow("Detected blob image", out)

cv2.waitKey(0)
cv2.destroyAllWindows()