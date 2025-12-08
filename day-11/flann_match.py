import cv2

# Read images
img2 = cv2.imread(r"../Images/Mercedes-Logo.png", cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread(r"../Images/Mercedes-car.png", cv2.IMREAD_GRAYSCALE)

resize_factor = 0.3
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

img1 = cv2.resize(img1, (int(w1*resize_factor), int(h1*resize_factor)))
img2 = cv2.resize(img2, (int(w2*resize_factor), int(h2*resize_factor)))

# 1. Create SIFT detector
sift = cv2.SIFT_create()

# 2. Detect keypoints & descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.30 * n.distance:
        good.append(m)

print("Good matches:", len(good))

result = cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("Sift/Flan Match", result)
cv2.imwrite("Outputs/flann_match.png", result)
cv2.waitKey(0)