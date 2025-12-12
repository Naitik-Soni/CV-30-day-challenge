import os
import re
import cv2
import numpy as np

# ---------- utilities ----------
def natural_key(text):
    """Key for natural/human sorting (numbers as integers)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', text)]

def get_sorted_file_paths(folder_path, ext=".png"):
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(ext)
    ]

    files_sorted = sorted(files, key=lambda x: natural_key(os.path.basename(x)))
    return files_sorted

# ---------- feature matching & homography ----------
def find_homography(img1_gray, img2_gray, min_matches=8):
    """Find homography mapping img2 -> img1 using ORB + BFMatcher. Returns H or None."""
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des2, des1, k=2)  # note: query=img2, train=img1

    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < min_matches:
        return None

    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)  # img2 points
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)  # img1 points

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

# ---------- panorama combine ----------
def warp_and_blend(base_img, new_img, H):
    """
    Warp new_img into base_img coordinate space using homography H (maps new_img -> base_img).
    Create a canvas that can contain both and blend overlap using linear feather.
    Returns combined image.
    """
    h1, w1 = base_img.shape[:2]
    h2, w2 = new_img.shape[:2]

    # corners of new_img
    corners_new = np.array([[0,0],[w2,0],[w2,h2],[0,h2]], dtype=np.float32).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_new, H).reshape(-1,2)

    # corners of base_img
    corners_base = np.array([[0,0],[w1,0],[w1,h1],[0,h1]], dtype=np.float32).reshape(-1,2)

    all_corners = np.vstack((warped_corners, corners_base))
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

    # translation to make everything positive
    tx = -x_min if x_min < 0 else 0
    ty = -y_min if y_min < 0 else 0

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    # Warp new_img onto canvas
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)
    H_translated = T @ H
    warped_new = cv2.warpPerspective(new_img, H_translated, (canvas_w, canvas_h))

    # Place base_img on canvas
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    x_offset = tx
    y_offset = ty
    canvas[y_offset:y_offset+h1, x_offset:x_offset+w1] = base_img

    # Create masks
    mask_base = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask_base[y_offset:y_offset+h1, x_offset:x_offset+w1] = 255
    mask_new = cv2.cvtColor(warped_new, cv2.COLOR_BGR2GRAY)
    _, mask_new = cv2.threshold(mask_new, 1, 255, cv2.THRESH_BINARY)

    # Regions
    overlap_mask = cv2.bitwise_and(mask_base, mask_new)
    only_base = cv2.bitwise_and(mask_base, cv2.bitwise_not(overlap_mask))
    only_new = cv2.bitwise_and(mask_new, cv2.bitwise_not(overlap_mask))

    # Start composing: base where only_base
    result = np.zeros_like(canvas)
    result[only_base==255] = canvas[only_base==255]
    result[only_new==255] = warped_new[only_new==255]

    # Blend overlap region with linear feathering along x (projected)
    if np.count_nonzero(overlap_mask) > 0:
        ys, xs = np.where(overlap_mask > 0)
        x_min_ov, x_max_ov = xs.min(), xs.max()
        width_ov = max(1, x_max_ov - x_min_ov)

        # compute alpha map from left->right in overlap region
        alpha = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        # linear ramp
        alpha[:, x_min_ov:x_max_ov+1] = np.linspace(0,1, width_ov+1)[None, :]

        # clamp alpha only inside overlap
        alpha = alpha * (overlap_mask.astype(np.float32)/255.0)

        # For overlap, use alpha*new + (1-alpha)*base
        for c in range(3):
            base_c = canvas[:,:,c].astype(np.float32)
            new_c = warped_new[:,:,c].astype(np.float32)
            blended = alpha * new_c + (1.0 - alpha) * base_c
            # apply only in overlap
            mask_idx = overlap_mask>0
            result[:,:,c][mask_idx] = blended[mask_idx].astype(np.uint8)

    return result

# ---------- main stitching function ----------
def stitch_images_in_folder(folder_path, ext=(".jpg", ".jpeg", ".png"), resize_factor=0.3, output_path="final_output.jpg"):
    """
    Sequentially stitch all images in 'folder_path' (natural order) into one panorama.
    All images are resized by resize_factor before processing.
    Saves final panorama to output_path and returns that path.
    """
    paths = get_sorted_file_paths(folder_path, ext=".png")
    if not paths:
        raise ValueError("No images found in folder with the given extensions.")

    # Load first image and resize
    first = cv2.imread(paths[0])
    if first is None:
        raise ValueError(f"Failed to read image: {paths[0]}")
    first = cv2.resize(first, (0,0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
    panorama = first.copy()
    print(f"[INFO] Starting with {os.path.basename(paths[0])} -> panorama size {panorama.shape[1]}x{panorama.shape[0]}")

    for idx, p in enumerate(paths[1:], start=2):
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] Skipping unreadable image: {p}")
            continue
        img = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

        # convert to gray for homography
        pan_gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        H = find_homography(pan_gray, img_gray)
        if H is None:
            print(f"[WARN] Homography not found between current panorama and {os.path.basename(p)} — skipping this image.")
            continue

        # warp and blend img into panorama
        panorama = warp_and_blend(panorama, img, H)
        cv2.imwrite(fr"Outputs/Panorama {idx}.jpg", panorama)
        print(f"[INFO] Stitched {os.path.basename(p)} -> current panorama {panorama.shape[1]}x{panorama.shape[0]}")

    # Save result
    cv2.imwrite(output_path, panorama)
    print(f"[DONE] Panorama saved to: {output_path} (size {panorama.shape[1]}x{panorama.shape[0]})")
    return output_path

# ---------- example usage ----------
if __name__ == "__main__":
    # change folder_path to your folder containing images 1..N
    folder_path = r"C:\Users\baps\Documents\Naitik Soni\ComputerVision\CV-30-day-challenge\Images\Panorama images"
    out = stitch_images_in_folder(folder_path, ext=(".jpg",), resize_factor=0.3, output_path=r"Outputs/final_output.jpg")
