import cv2
import numpy as np

# --------- 1. Get ORB descriptors ----------
img = cv2.imread(r"../Images/deer.jpg", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(img, None)

print("Number of keypoints:", len(keypoints))
print("Descriptors shape:", descriptors.shape)   # (N, 32)

# Pick a single descriptor (e.g., first one)
desc = descriptors[0]  # shape: (32,), dtype: uint8

# --------- 2. Convert descriptor to 256-bit binary string ----------
# Each element is a byte -> 8 bits
bit_string = ''.join(f'{byte:08b}' for byte in desc)  # length = 256

print("\nRaw 256-bit descriptor string:")
print(bit_string)
print("Length:", len(bit_string))  # should be 256

# Optional: group bits by byte (for clarity)
print("\nDescriptor grouped by bytes (index: bits):")
for i, byte in enumerate(desc):
    print(f"{i:02d}: {byte:08b}")

# --------- 3. Visualize as a 16x16 pattern ----------
rows, cols = 32, 8  # 16 * 16 = 256 bits

print("\n16x16 binary pattern (█ = 1, . = 0):")
for r in range(rows):
    row_bits = bit_string[r * cols : (r + 1) * cols]
    # Use block char for 1 and dot for 0
    visual_row = ''.join('█' if b == '1' else '.' for b in row_bits)
    print(visual_row, "  ", row_bits)