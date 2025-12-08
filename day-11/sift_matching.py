import cv2

# Read images
img2 = cv2.imread(r"../Images/atom_ro.webp", cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread(r"../Images/atom_og.jpg", cv2.IMREAD_GRAYSCALE)

resize_factor = 0.7
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

img1 = cv2.resize(img1, (int(w1*resize_factor), int(h1*resize_factor)))
img2 = cv2.resize(img2, (int(w2*resize_factor), int(h2*resize_factor)))

# 1. Create SIFT detector
sift = cv2.SIFT_create()

# 2. Detect keypoints & descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 3. BFMatcher (for float descriptors)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# 4. KNN match
matches = bf.knnMatch(des1, des2, k=2)

# 5. Lowe Ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good_matches.append(m)

# 6. Draw
result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("SIFT Match", result)
cv2.imwrite("Outputs/sift_match.png", result)
cv2.waitKey(0)