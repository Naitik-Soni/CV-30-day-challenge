import cv2
import numpy as np
from panorama_utils import *

# ---------- robust get_transformed_image ----------
def get_transformed_image(image_to_warp, H, canvas_width, canvas_height=None, x_offset=0, y_offset=0):
    """
    Warp image_to_warp using homography H into a canvas of width `canvas_width`.
    Optionally provide canvas_height (defaults to image_to_warp height),
    and x_offset/y_offset: how much to translate the warped result to the right/down.

    NOTE: H is expected to map points from image_to_warp -> destination coordinate space.
    We apply a translation T so the warped content appears shifted by (x_offset, y_offset).
    """
    if H is None:
        raise ValueError("Homography is None in get_transformed_image")

    h2, w2 = image_to_warp.shape[:2]
    if canvas_height is None:
        canvas_height = h2

    # Translation matrix to shift the warped image inside the output canvas
    T = np.array([[1, 0, x_offset],
                  [0, 1, y_offset],
                  [0, 0, 1]], dtype=np.float64)

    H_shifted = T @ H  # apply H then translate (i.e. translate after warp)
    canvas_w = int(canvas_width)
    canvas_h = int(canvas_height)

    # Warp the image into the canvas (supports larger canvas)
    warped = cv2.warpPerspective(image_to_warp, H_shifted, (canvas_w, canvas_h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    return warped

# ---------- safer get_homography_image ----------
def get_homography_image(prev_img, img2_path, final_width, resize_factor):
    """
    prev_img: already-loaded & resized image (base)
    img2_path: path to next image to load & warp into base coordinate system
    final_width: desired output canvas width (in same resized units)
    resize_factor: applied during load_image
    Returns: warped image on a canvas (width=final_width) or None if H failed
    """
    # Load next image (resized) -- ensure load_image uses same resize_factor
    image2 = load_image(img2_path, resize=True, resize_factor=resize_factor)  # adjust to your helper signature

    if image2 is None:
        print(f"[WARN] couldn't load {img2_path}")
        return None

    # Compute features/descriptors
    kp1, des1 = get_features(prev_img)
    kp2, des2 = get_features(image2)

    if des1 is None or des2 is None:
        print("[WARN] descriptors missing for images; skipping")
        return None

    good_matches = get_good_matches(des1, des2)   # ensure your function expects (des1, des2) in that order
    if not good_matches or len(good_matches) < 8:
        print(f"[WARN] not enough matches ({0 if not good_matches else len(good_matches)}); skipping {img2_path}")
        return None

    src_kp, dst_kp = get_final_points(kp1, kp2, good_matches)
    if len(src_kp) < 4 or len(dst_kp) < 4:
        print("[WARN] not enough matched keypoints after filtering; skipping")
        return None

    # IMPORTANT: decide order: if get_final_points returned (kp_from_prev, kp_from_new),
    # then H = findHomography(src=kp_from_new, dst=kp_from_prev) mapping new->prev
    # Here we assume src_kp are points in prev_img and dst_kp are points in image2
    # Adjust the following depending on your helper's exact return order.
    # I'll assume get_final_points(kp1,kp2, matches) -> (pts1, pts2) where pts1 correspond to kp1 (prev), pts2 -> kp2 (new)
    pts_prev = np.array(src_kp, dtype=np.float32).reshape(-1,2)
    pts_new  = np.array(dst_kp, dtype=np.float32).reshape(-1,2)

    # We want H that maps points from new -> prev (so we can warp new into prev coordinate space)
    H, mask = cv2.findHomography(pts_new, pts_prev, cv2.RANSAC, 5.0)

    if H is None:
        print("[WARN] findHomography returned None; skipping")
        return None

    # Compute an x_offset so the warped image sits to the right side of the canvas.
    # If final_width is current total canvas width (accumulating w), we want to translate the warped image
    # so its content does not overlap the leftmost region. A safe offset is (final_width - image2_width)
    # but we must clamp to >=0.
    h2, w2 = image2.shape[:2]
    x_offset = max(0, int(final_width - w2))
    canvas_h = max(prev_img.shape[0], h2)
    warped_canvas = get_transformed_image(image2, H, canvas_width=final_width, canvas_height=canvas_h, x_offset=x_offset, y_offset=0)

    return warped_canvas

# ---------- robust get_panorama_image ----------
def get_panorama_image(w, h, homography_images):
    """
    Combines the list of warped canvases into one panorama.
    Each element in homography_images is a canvas where the new image has been warped into a canvas
    of width final_width for that step. We will progressively merge by creating a sufficiently large canvas
    and blending overlaps.
    """
    if not homography_images:
        return None

    # Determine max width and height among canvases
    widths = [img.shape[1] for img in homography_images if img is not None]
    heights = [img.shape[0] for img in homography_images if img is not None]

    canvas_w = max(widths)
    canvas_h = max(heights)

    # Start with a black canvas
    panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place first image at left (assuming first image has content starting at x=0)
    panorama[0:homography_images[0].shape[0], 0:homography_images[0].shape[1]] = homography_images[0]

    for i in range(1, len(homography_images)):
        img = homography_images[i]
        if img is None:
            continue

        # find non-black mask for this img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # compute overlap with existing panorama
        pano_gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        _, pano_mask = cv2.threshold(pano_gray, 1, 255, cv2.THRESH_BINARY)

        overlap = cv2.bitwise_and(mask, pano_mask)

        # place non-overlapping new areas directly
        only_new = cv2.bitwise_and(mask, cv2.bitwise_not(overlap))
        ys, xs = np.where(only_new > 0)
        if ys.size > 0 and xs.size > 0:
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            panorama[y0:y1+1, x0:x1+1][only_new[y0:y1+1, x0:x1+1]==255] = img[y0:y1+1, x0:x1+1][only_new[y0:y1+1, x0:x1+1]==255]

        # blend overlap area using linear alpha along x
        if np.count_nonzero(overlap) > 0:
            ys, xs = np.where(overlap > 0)
            x_min, x_max = xs.min(), xs.max()
            width = max(1, x_max - x_min)
            alpha_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)
            alpha_map[:, x_min:x_max+1] = np.linspace(0, 1, width+1)[None, :]

            for c in range(3):
                pano_c = panorama[:,:,c].astype(np.float32)
                img_c = img[:,:,c].astype(np.float32)
                blended = (1.0 - alpha_map) * pano_c + alpha_map * img_c
                # write blended only where overlap
                mask_idx = overlap > 0
                panorama[:,:,c][mask_idx] = blended[mask_idx].astype(np.uint8)

    return panorama