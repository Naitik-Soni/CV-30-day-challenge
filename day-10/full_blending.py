import cv2
import numpy as np
import os # Included for file path check instructions

# --- CORE PYRAMID FUNCTIONS ---

def build_pyramid(image, levels=5):
    """Builds the Gaussian and Laplacian Pyramids for an image."""
    # Use float32 for all pyramid calculations to avoid data loss/clipping
    G = image.copy().astype(np.float32) 
    gp = [G]
    lp = []
    
    for i in range(levels):
        # 1. Gaussian Down (smoothing + downsampling)
        G = cv2.pyrDown(G)
        gp.append(G)

        # 2. Gaussian Up (upsampling for difference calculation)
        # Ensure the upsampled image size exactly matches the previous level's size
        G_up = cv2.pyrUp(G, dstsize=(gp[i].shape[1], gp[i].shape[0]))
        
        # 3. Laplacian (Difference: Current Gaussian - Upsampled Next Gaussian)
        L = cv2.subtract(gp[i], G_up)
        lp.append(L)
        
    return gp, lp

def build_gaussian_mask_pyramid(mask, levels=5):
    """Builds the Gaussian Pyramid for the mask (used as blending weights)."""
    # The mask must be float32 for interpolation and multiplication
    G = mask.copy().astype(np.float32)
    gpM = [G]
    
    for i in range(levels):
        # cv2.pyrDown handles the necessary blurring and downsampling
        G = cv2.pyrDown(G)
        gpM.append(G)
        
    return gpM

def create_mask(image_height, image_width):
    """Creates a binary mask (H x W) for a corner split (Top-Left White)."""
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    h_mid = image_height // 2
    w_mid = image_width // 2
    # White (255) in the top-left quarter
    mask[0:h_mid, 0:w_mid] = 255
    return mask

def blend_pyramids(lpA, lpB, gpA, gpB, gpM):
    """
    Blends two Laplacian Pyramids using a Gaussian Mask Pyramid as weights.
    
    Args:
        lpA, lpB: Laplacian pyramids for Image A and Image B details.
        gpA, gpB: Gaussian pyramids for the coarsest (top) level blend.
        gpM: Gaussian pyramid for the mask weights.
    """
    LS = []
    
    # 1. Blend the Laplacian levels (detail layers)
    # gpM[:-1] uses all Gaussian mask levels except the coarsest one
    for la, lb, gm in zip(lpA, lpB, gpM[:-1]):
        # Normalize mask to 0-1 range and ensure 3 channels for element-wise multiplication
        gm_normalized = gm / 255.0
        gm_3channel = cv2.merge([gm_normalized, gm_normalized, gm_normalized])
        
        # Blending formula: L_S = G_M * L_A + (1 - G_M) * L_B
        ls = np.add(np.multiply(la, gm_3channel), np.multiply(lb, 1.0 - gm_3channel))
        LS.append(ls)
        
    # 2. Blend the Coarsest (Top) Gaussian level (the low-frequency base)
    top_A = gpA[-1]
    top_B = gpB[-1]
    top_M = gpM[-1] # Use the smallest Gaussian mask level
    
    # Normalize top mask and ensure 3 channels
    top_M_normalized = top_M / 255.0
    top_M_3channel = cv2.merge([top_M_normalized, top_M_normalized, top_M_normalized])
    
    # Blend the top level Gaussians
    top_S = np.add(np.multiply(top_A, top_M_3channel), np.multiply(top_B, 1.0 - top_M_3channel))
    LS.append(top_S)

    return LS

def reconstruct(blended_pyramid):
    """Reconstructs the final image from the Blended Laplacian Pyramid."""
    # Start with the coarsest (top) level
    current = blended_pyramid[-1]
    
    # Iterate backwards through the remaining detail levels
    for i in range(len(blended_pyramid) - 2, -1, -1):
        # PyrUp the current image (coarse)
        current_up = cv2.pyrUp(current, dstsize=(blended_pyramid[i].shape[1], blended_pyramid[i].shape[0]))
        # Add the detail from the next Laplacian level
        current = cv2.add(current_up, blended_pyramid[i])
        
    # Convert back to 8-bit image and clip values (0-255)
    return np.clip(current, 0, 255).astype(np.uint8)


# --- FULL BLENDING EXECUTION PIPELINE ---
if __name__ == '__main__':
    
    # === STEP 1: LOAD AND RESIZE IMAGES ===
    
    # --------------------------------------------------------------------------------------------------
    # NOTE FOR USER:
    # 
    # REPLACE THE FOLLOWING PLACEHOLDER CODE with your actual image loading.
    # The mask file 'mask.png' is not required as it will be generated dynamically.
    # 
    # Example:
    imgA = cv2.imread(r'../Images/deer.jpg')
    imgB = cv2.imread(r'../Images/Autumn.jpg')
    # --------------------------------------------------------------------------------------------------
    
    # Placeholder Image A (e.g., Green/Yellow, Size 500x400)
    # imgA = np.zeros((500, 400, 3), dtype=np.uint8)
    # imgA[:, :, 1] = 200 # Green channel
    # imgA[100:400, 100:300, 2] = 255 # Yellow square
    
    # Placeholder Image B (e.g., Blue/Red, Size 600x500)
    # imgB = np.zeros((600, 500, 3), dtype=np.uint8)
    # imgB[:, :, 0] = 200 # Blue channel
    # imgB[150:450, 150:350, 2] = 255 # Red square
    
    # Use the larger image (Image B) as the target dimension for consistency.
    H_target, W_target, _ = imgB.shape
    
    # Resize Image A to match Image B's dimensions (critical step for different-sized inputs)
    imgA_resized = cv2.resize(imgA, (W_target, H_target), interpolation=cv2.INTER_LINEAR)
    
    print(f"Original A size: {imgA.shape}, Original B size: {imgB.shape}")
    print(f"Resized A and B target size: {W_target}x{H_target}")
    
    # === STEP 2: CREATE MASK AND PYRAMIDS ===
    N_LEVELS = 5

    # Create the binary mask using the target dimensions
    binary_mask = create_mask(H_target, W_target)
    
    # Build the required pyramids
    gpA, lpA = build_pyramid(imgA_resized, N_LEVELS)
    gpB, lpB = build_pyramid(imgB, N_LEVELS)
    gpM = build_gaussian_mask_pyramid(binary_mask, N_LEVELS)
    
    # === STEP 3: BLEND AND RECONSTRUCT ===

    # Blend the pyramids using the full function signature
    blended_laplacian = blend_pyramids(lpA, lpB, gpA, gpB, gpM)

    # Reconstruct the final image from the blended pyramid
    seamless_blend = reconstruct(blended_laplacian)

    # === STEP 4: DISPLAY RESULTS ===
    
    # Create a simple non-blended composite for comparison (sharp seam)
    # The mask must be extended to 3 channels to use with np.where
    mask_3d = np.stack([binary_mask]*3, axis=2)
    simple_composite = np.where(mask_3d == 255, imgA_resized, imgB)
    
    # Concatenate the inputs, sharp composite, and seamless blend for comparison
    comparison_stack = np.vstack([
        np.hstack([imgA_resized, simple_composite]),
        np.hstack([imgB, seamless_blend])
    ])

    print("Blending complete. Showing comparison window.")
    cv2.imshow('Multi-Resolution Blending Result', comparison_stack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()