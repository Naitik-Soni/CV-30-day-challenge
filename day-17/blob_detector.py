import cv2

image_path = r"C:\Users\baps\Documents\Naitik Soni\ComputerVision\CV-30-day-challenge\Images\night_sky.jpg"

image = cv2.imread(image_path)

img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 0.1
params.maxArea = 5000

params.filterByCircularity = True
params.minCircularity = 0.1

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(img)

img_with_blobs = cv2.drawKeypoints(
    image, keypoints, None,
    (0,255,0),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imshow("Blobs", img_with_blobs)
cv2.waitKey(0)
