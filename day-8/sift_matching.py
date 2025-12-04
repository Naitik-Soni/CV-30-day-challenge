import cv2
import numpy as np

# -------------------- Load images --------------------
# img1 = scene (car), img2 = object (logo)
img1 = cv2.imread(r"../Images/Mercedes-car.png", 0)    # scene
img2 = cv2.imread(r"../Images/Mercedes-logo.png", 0)   # logo

if img1 is None or img2 is None:
    raise ValueError("Error loading images. Check the file paths.")

# -------------------- Resize (optional) --------------------
resize_Factor = 0.5
h1, w1 = img1.shape
img1 = cv2.resize(img1, (int(w1 * resize_Factor), int(h1 * resize_Factor)))

resize_factor2 = 0.2
h2, w2 = img2.shape
img2 = cv2.resize(img2, (int(w2 * resize_factor2), int(h2 * resize_factor2)))

# -------------------- SIFT keypoints & descriptors --------------------
sift = cv2.SIFT_create()

# kp1/des1 -> car, kp2/des2 -> logo
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

print("Keypoints in car image:", len(kp1))
print("Keypoints in logo image:", len(kp2))

if des1 is None or des2 is None:
    raise ValueError("No descriptors found. Try changing resize factors or image quality.")

# -------------------- Matcher (KNN + Lowe's ratio) --------------------
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# IMPORTANT: query = logo (des2), train = car (des1)
matches = bf.knnMatch(des2, des1, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:   # Lowe's ratio test
        good.append(m)

print("Good matches:", len(good))

MIN_MATCH_COUNT = 10

# -------------------- Homography + Bounding Box --------------------
if len(good) >= MIN_MATCH_COUNT:
    # src_pts: points in logo image (object)
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    # dst_pts: points in car image (scene)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # RANSAC to find robust homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is not None:
        # Corners of logo image
        h_logo, w_logo = img2.shape
        pts = np.float32([[0, 0],
                          [w_logo, 0],
                          [w_logo, h_logo],
                          [0, h_logo]]).reshape(-1, 1, 2)

        # Project logo corners into car image
        dst = cv2.perspectiveTransform(pts, M)

        # Draw bounding box on car image
        car_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        cv2.polylines(car_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

        # Optional: draw inlier matches for visualization
        matches_mask = mask.ravel().tolist()
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=matches_mask,
            flags=cv2.DrawMatchesFlags_DEFAULT
        )

        match_vis = cv2.drawMatches(
            img2, kp2,        # logo on left
            img1, kp1,        # car on right
            good, None, **draw_params
        )

        cv2.imshow("Detected Logo (Bounding Box on Car)", car_color)
        cv2.imshow("Inlier Matches (Logo ↔ Car)", match_vis)
    else:
        print("Homography could not be computed.")
        cv2.imshow("Car Image", img1)

else:
    print(f"Not enough good matches: {len(good)}/{MIN_MATCH_COUNT}")
    cv2.imshow("Car Image (No Detection)", img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
