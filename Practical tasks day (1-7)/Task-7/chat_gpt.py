import cv2
import numpy as np
import os

# ----------------------------
# Utility Functions
# ----------------------------
def resize_fixed_width(img, width=900):
    h, w = img.shape[:2]
    scale = width / float(w)
    new_h = int(h * scale)
    return cv2.resize(img, (width, new_h)), scale

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def warp_plate(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


# ----------------------------
# Main Plate Detection Function
# ----------------------------
def detect_plate(image_path, debug_prefix="debug"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found: " + image_path)

    os.makedirs(".", exist_ok=True)

    # 1) Resize for stability
    resized, scale = resize_fixed_width(img, width=900)
    rh, rw = resized.shape[:2]
    img_area = rh * rw

    # 2) Preprocessing
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Reduce noise but keep edges
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 3) Vertical edges (number plates have vertical strokes)
    sobelX = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    _, th = cv2.threshold(sobelX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4) Morphological close → join plate characters
    k = max(3, rw // 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k * 6 + 1, k + 1))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5) Remove small blobs
    cleaned = morph.copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph, connectivity=8)
    min_area = img_area // 800

    cleaned = np.zeros_like(morph)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            cleaned[labels == i] = 255

    # 6) Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_img = resized.copy()

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.0008:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / float(h)

        rectangularity = area / (w * h + 1e-6)
        solidity = area / (cv2.contourArea(cv2.convexHull(cnt)) + 1e-6)

        # Choose acceptable plate geometry
        if 2.0 < aspect < 6.5 and rectangularity > 0.45 and solidity > 0.5:
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype("float32")
            else:
                # Use minAreaRect if approx not 4-sided
                rect = cv2.minAreaRect(cnt)
                pts = cv2.boxPoints(rect).astype("float32")

            candidates.append((pts, area))
            cv2.drawContours(debug_img, [pts.astype(int)], -1, (0, 255, 0), 2)

    cv2.imwrite(f"{debug_prefix}_debug.png", debug_img)

    if not candidates:
        print("❌ No plate candidates found!")
        return []

    # Sort by largest area first (plate usually largest rectangle)
    candidates.sort(key=lambda x: x[1], reverse=True)

    final_crops = []

    for idx, (pts, area) in enumerate(candidates[:5]):
        crop = warp_plate(resized, pts)

        if crop is None:
            continue

        wh_ratio = crop.shape[1] / float(crop.shape[0])
        if 1.3 < wh_ratio < 8.0:
            final_crops.append(crop)
            cv2.imwrite(f"plate_crop_{idx}.png", crop)

    if final_crops:
        print(f"✅ Extracted {len(final_crops)} plate crop(s). Saved: plate_crop_*.png")
    else:
        print("❌ Candidates found but not valid after warping.")

    return final_crops


# ---------------------------------------
# Run
# ---------------------------------------
if __name__ == "__main__":
    detect_plate("car-2.jpeg", debug_prefix="plate_debug")
