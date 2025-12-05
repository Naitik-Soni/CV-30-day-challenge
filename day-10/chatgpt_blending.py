"""
blending_examples.py

Requirements:
    pip install opencv-python numpy

Usage:
    python blending_examples.py

Make sure you provide two images of roughly the same size, or let the script resize them.
"""

import cv2
import numpy as np
import os

def load_and_resize(path, size=(600, 600)):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.resize(img, size)

def alpha_blend(A, B, alpha=0.5):
    """Simple weighted average blend."""
    return cv2.addWeighted(A, alpha, B, 1 - alpha, 0)

def mask_alpha_blend(A, B, mask):
    """
    Blend using a mask with values in [0,1] float32.
    result = mask*A + (1-mask)*B
    mask should be single channel float32 or 3-channel float32.
    """
    if mask.ndim == 2:
        mask = mask[:, :, None]
    return (mask * A.astype(np.float32) + (1 - mask) * B.astype(np.float32)).astype(np.uint8)

def linear_gradient_mask(shape, horizontal=True, invert=False):
    """Create a float mask that transitions linearly from 1->0 across width (or height)."""
    h, w = shape[:2]
    if horizontal:
        grad = np.linspace(1.0, 0.0, w)
        if invert:
            grad = grad[::-1]
        mask = np.tile(grad, (h, 1)).astype(np.float32)
    else:
        grad = np.linspace(1.0, 0.0, h)
        if invert:
            grad = grad[::-1]
        mask = np.tile(grad[:, None], (1, w)).astype(np.float32)
    return mask

def feather_mask(shape, center=None, radius=None):
    """Creates a circular feathered mask (1 inside center, 0 outside), with gaussian feathering."""
    h, w = shape[:2]
    if center is None:
        center = (w//2, h//2)
    if radius is None:
        radius = min(h, w) // 3
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = np.clip((radius - dist) / radius, 0, 1)
    # smooth edges with Gaussian blur for better feather
    mask = cv2.GaussianBlur(mask.astype(np.float32), (31,31), 0)
    return mask.astype(np.float32)

def pyramid_blend(A, B, mask, levels=6):
    """
    Multi-resolution blending using Laplacian pyramids.
    - A, B: color images (uint8) same size
    - mask: single channel float32 in [0,1] same size
    - levels: number of pyramid levels
    """
    # Convert mask to 3-channel float
    if mask.ndim == 2:
        mask3 = np.stack([mask]*3, axis=2)
    else:
        mask3 = mask.astype(np.float32)

    # Build Gaussian pyramids for A, B and mask
    GA = [A.astype(np.float32)]
    GB = [B.astype(np.float32)]
    GM = [mask3.astype(np.float32)]
    for i in range(levels):
        GA.append(cv2.pyrDown(GA[-1]))
        GB.append(cv2.pyrDown(GB[-1]))
        GM.append(cv2.pyrDown(GM[-1]))

    # Build Laplacian pyramids for A and B
    LA = []
    LB = []
    for i in range(levels):
        size = (GA[i].shape[1], GA[i].shape[0])  # width, height
        GE_A_up = cv2.pyrUp(GA[i+1], dstsize=size)
        GE_B_up = cv2.pyrUp(GB[i+1], dstsize=size)
        LA.append(cv2.subtract(GA[i], GE_A_up))
        LB.append(cv2.subtract(GB[i], GE_B_up))
    # The smallest level
    LA.append(GA[-1])
    LB.append(GB[-1])

    # Blend pyramids using mask's Gaussian pyramid
    LS = []
    for la, lb, gm in zip(LA, LB, GM):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # Reconstruct
    result = LS[-1]
    for i in range(len(LS)-2, -1, -1):
        size = (LS[i].shape[1], LS[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size)
        result = cv2.add(result, LS[i])

    # Clip and convert
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def seamless_clone(A, B, mask):
    """
    Use OpenCV Poisson seamless cloning to copy A into B guided by mask.
    - A: source patch (uint8)
    - B: destination image (uint8)
    - mask: single channel uint8 mask where 255 = copy region
    We will place A at the center of B for demo.
    """
    h, w = B.shape[:2]
    center = (w//2, h//2)
    if mask.dtype != np.uint8:
        m = (mask * 255).astype(np.uint8)
    else:
        m = mask
    try:
        output = cv2.seamlessClone(A, B, m, center, cv2.NORMAL_CLONE)
    except Exception as e:
        print("seamlessClone failed:", e)
        return B
    return output

def show_and_save(name, img, folder="Outputs"):
    os.makedirs("Outputs", exist_ok=True)
    path = os.path.join(folder, f"{name}.jpg")
    cv2.imwrite(path, img)
    cv2.imshow(name, img)

def ensure_same_size(A, B, target_size=(600,600)):
    A_rs = cv2.resize(A, target_size)
    B_rs = cv2.resize(B, target_size)
    return A_rs, B_rs

def main():
    # -------- CONFIG --------
    img1_path = r"../Images/full moon.jpg"   # replace with your path
    img2_path = r"../Images/deer.jpg"   # replace with your path
    size = (400, 400)          # resize target (w,h)
    levels = 6                 # pyramid levels
    # ------------------------

    # Load
    A = load_and_resize(img1_path, size=size)
    B = load_and_resize(img2_path, size=size)

    # Ensure same size (height,width)
    A, B = ensure_same_size(A, B, size)

    # 1) Simple alpha blending
    alpha_res = alpha_blend(A, B, alpha=0.6)
    show_and_save("alpha_blend", alpha_res)

    # 2) Linear gradient blend (left->right)
    grad_mask = linear_gradient_mask(A.shape, horizontal=True)  # 1 on left, 0 on right
    linear_res = mask_alpha_blend(A, B, grad_mask)
    show_and_save("linear_gradient_blend", linear_res)

    # 3) Feathered circular mask blend
    fmask = feather_mask(A.shape)
    feather_res = mask_alpha_blend(A, B, fmask)
    show_and_save("feather_blend", feather_res)

    # 4) Pyramid (multi-resolution) blend using same linear mask but smoothed
    # use the linear mask but blur it to avoid hard seams in pyramid
    mask_for_pyr = cv2.GaussianBlur(grad_mask, (51,51), 0)
    pyr_res = pyramid_blend(A, B, mask_for_pyr, levels=levels)
    show_and_save("pyramid_blend", pyr_res)

    # 5) Poisson seamless clone
    # create binary mask for the left half of A
    half_mask = np.zeros((A.shape[0], A.shape[1]), dtype=np.uint8)
    half_mask[:, :A.shape[1]//2] = 255
    # For clone, we need source patch; put A on top of B's region by cropping or using A directly.
    # Here we clone A into B using the half_mask (centered)
    clone_res = seamless_clone(A, B, half_mask)
    show_and_save("seamless_clone", clone_res)

    # Arrange a combined comparison image (grid)
    h, w = A.shape[:2]
    row1 = np.hstack([A, B, alpha_res])
    row2 = np.hstack([linear_res, feather_res, pyr_res])
    # resize clone to same h,w for layout if needed
    clone_small = cv2.resize(clone_res, (w, h))
    row3 = np.hstack([clone_small, np.zeros_like(A), np.zeros_like(A)])  # padding
    # Stack rows (ensure same width)
    comp = np.vstack([row1, row2, row3])

    show_and_save("comparison_grid", comp)

    print("Displaying windows. Press any key in a window to close all and finish.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Outputs saved to ./outputs/")

if __name__ == "__main__":
    main()