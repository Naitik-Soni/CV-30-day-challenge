import cv2

# --- Laplacian Pyramid Example ---
G1 = cv2.imread(r"./Outputs/G1.png")  # Read the first Gaussian level
G2 = cv2.imread(r"./Outputs/G2.png")  # Read the second Gaussian level
Original = cv2.imread(r"./Outputs/Original.png")  # Read the original image

# 1. Upsample the coarser Gaussian image (G2)
G2_up = cv2.pyrUp(G2, dstsize=(G1.shape[1], G1.shape[0])) # Force size match for subtraction

# 2. Calculate the Laplacian (L1)
# Use cv2.subtract to handle potential negative values/clipping in image data
# For visual display, we often need to convert to a suitable type (e.g., 32-bit float)
L1 = cv2.subtract(G1, G2_up)

print(f"Laplacian Level 1 size (Detail image): {L1.shape}")

# 1. Upsample the coarser Gaussian image (G1)
G1_up = cv2.pyrUp(G1, dstsize=(Original.shape[1], Original.shape[0])) # Force size match for subtraction

# 2. Calculate the Laplacian (L2)
# Use cv2.subtract to handle potential negative values/clipping in image data
# For visual display, we often need to convert to a suitable type (e.g., 32-bit float)
L2 = cv2.subtract(Original, G1_up)

print(f"Laplacian Level 2 size (Detail image): {L2.shape}")

# L1 contains the high-frequency details lost when going from G1 to G2.
# A full Laplacian Pyramid would also include the smallest Gaussian image (G2) as its top level.

# Save the Laplacian images for reference
cv2.imwrite(r"./Outputs/L1.png", L1)
cv2.imwrite(r"./Outputs/L2.png", L2)
