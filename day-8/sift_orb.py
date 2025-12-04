import cv2
import numpy as np

# --------- CONFIG ---------
IMG1_PATH = r"..\Images\atom_og.jpg"
IMG2_PATH = r"..\Images\ATOM_ROTATED.webp"
MAX_MATCHES_TO_DRAW = 50      # show top N matches
SIFT_RATIO = 0.75             # Lowe's ratio for SIFT
ORB_RATIO = 0.8               # a bit more relaxed for ORB
OUTPUT_PATH = "sift_orb_matches_comparison.jpg"
# --------------------------


def get_good_matches_knn(des1, des2, matcher, ratio):
    """
    Apply KNN + Lowe's ratio test and return list of good matches.
    """
    knn_matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m_n in knn_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def draw_matches(img1, kp1, img2, kp2, matches, title_text=""):
    """
    Draw matches between two images using cv2.drawMatches.
    """
    vis = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    if title_text:
        cv2.putText(
            vis, title_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
        )
    return vis


def main():
    # 1. Load images (color for visualization)
    img1_color = cv2.imread(IMG1_PATH)
    img2_color = cv2.imread(IMG2_PATH)

    resize_factor = 0.5
    img1_color = cv2.resize(img1_color, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
    img2_color = cv2.resize(img2_color, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

    if img1_color is None or img2_color is None:
        print("Error: Could not load one or both images.")
        return

    # Convert to grayscale for feature detection/descriptors
    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    # 2. SIFT
    sift = cv2.SIFT_create()
    kp1_s, des1_s = sift.detectAndCompute(img1_gray, None)
    kp2_s, des2_s = sift.detectAndCompute(img2_gray, None)

    print(f"[SIFT] keypoints in img1: {len(kp1_s)}, img2: {len(kp2_s)}")

    matcher_s = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    good_s = get_good_matches_knn(des1_s, des2_s, matcher_s, SIFT_RATIO)
    # sort by distance (best first)
    good_s = sorted(good_s, key=lambda m: m.distance)
    print(f"[SIFT] good matches after ratio test: {len(good_s)}")

    # 3. ORB
    orb = cv2.ORB_create()
    kp1_o, des1_o = orb.detectAndCompute(img1_gray, None)
    kp2_o, des2_o = orb.detectAndCompute(img2_gray, None)

    if des1_o is None or des2_o is None:
        print("[ORB] No descriptors found in one of the images.")
        return

    print(f"[ORB] keypoints in img1: {len(kp1_o)}, img2: {len(kp2_o)}")

    matcher_o = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    good_o = get_good_matches_knn(des1_o, des2_o, matcher_o, ORB_RATIO)
    good_o = sorted(good_o, key=lambda m: m.distance)
    print(f"[ORB] good matches after ratio test: {len(good_o)}")

    # 4. Draw top matches for visualization
    sift_vis = draw_matches(
        img1_color, kp1_s,
        img2_color, kp2_s,
        good_s[:MAX_MATCHES_TO_DRAW],
        title_text=f"SIFT (good matches: {len(good_s)})"
    )

    orb_vis = draw_matches(
        img1_color, kp1_o,
        img2_color, kp2_o,
        good_o[:MAX_MATCHES_TO_DRAW],
        title_text=f"ORB (good matches: {len(good_o)})"
    )

    # 5. Stack SIFT vs ORB side by side
    # Resize to same height if needed
    h1, w1 = sift_vis.shape[:2]
    h2, w2 = orb_vis.shape[:2]

    if h1 != h2:
        # resize orb_vis to sift_vis height
        scale = h1 / h2
        orb_vis = cv2.resize(orb_vis, (int(w2 * scale), h1))

    comparison = np.hstack([sift_vis, orb_vis])

    # 6. Show and save
    cv2.imshow("SIFT vs ORB - Feature Matching", comparison)
    cv2.imwrite(OUTPUT_PATH, comparison)
    print(f"Saved comparison image to: {OUTPUT_PATH}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
