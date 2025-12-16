import cv2
import numpy as np

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2)

def get_features(img):
    return sift.detectAndCompute(img, None)

def match_features(des1, des2):
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def stitch_images(img1, img2):
    kp1, des1 = get_features(img1)
    kp2, des2 = get_features(img2)

    if des1 is None or des2 is None:
        return None

    good = match_features(des1, des2)

    if len(good) < 10:
        return None

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None

    # Create canvas
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas = cv2.warpPerspective(img2, H, (w1 + w2, h1))
    cv2.imshow("Canvas pano", canvas)
    cv2.waitKey(0)
    canvas[0:h1, 0:w1] = img1

    return canvas

def get_panorama(images):
    panorama = images[0]

    for i in range(1, len(images)):
        print(f"\nStitching image {i+1}/{len(images)}")

        result = stitch_images(panorama, images[i])

        if result is None:
            print("❌ Stitch FAILED (no homography / overlap)")
            continue
        else:
            print("✅ Stitch SUCCESS")

        panorama = result

    return panorama

