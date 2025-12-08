import cv2

# Read images
img2 = cv2.imread(r"../Images/atom_ro.webp", cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread(r"../Images/atom_og.jpg", cv2.IMREAD_GRAYSCALE)

resize_factor = 0.7
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

img1 = cv2.resize(img1, (int(w1*resize_factor), int(h1*resize_factor)))
img2 = cv2.resize(img2, (int(w2*resize_factor), int(h2*resize_factor)))


orb = cv2.ORB_create(nfeatures=1000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
cv2.imshow("ORB Match", result)
cv2.imwrite("Outputs/orb_match.png", result)
cv2.waitKey(0)