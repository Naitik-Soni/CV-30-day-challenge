import cv2
import numpy as np

# ============ 1. Load image and compute ORB descriptors ============
img = cv2.imread(r"../Images/deer.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Could not load image")

h, w = img.shape
orb = cv2.ORB_create(nfeatures=1000)
keypoints, descriptors = orb.detectAndCompute(img, None)

print("Total keypoints:", len(keypoints))
print("Descriptors shape:", descriptors.shape)  # (N, 32)

# ============ 2. Helper: convert descriptor row -> 256 bits & pattern ============

def desc_to_bitstring(desc_row):
    """desc_row: shape (32,), uint8 -> '0101...'(256 bits)"""
    return ''.join(f'{byte:08b}' for byte in desc_row)

def print_descriptor_pattern(name, kp, desc_row):
    """
    Print:
    - name label
    - keypoint location
    - 16x16 visual pattern (█ for 1, . for 0)
    """
    bit_string = desc_to_bitstring(desc_row)  # 256 bits
    print(f"\n=== {name} ===")
    print(f"Location (x, y): ({kp.pt[0]:.1f}, {kp.pt[1]:.1f})")
    print(f"Raw bits (first 64): {bit_string[:64]}...")

    rows, cols = 16, 16
    print("16x16 pattern (█ = 1, . = 0):")
    for r in range(rows):
        row_bits = bit_string[r * cols : (r + 1) * cols]
        visual_row = ''.join('█' if b == '1' else '.' for b in row_bits)
        print(visual_row, "  ", row_bits)

# ============ 3. Pick keypoints from different image regions ============

# Divide image into a 3x3 grid, pick one keypoint per cell (if available)
num_rows, num_cols = 3, 3
selected = []  # list of (label, keypoint, descriptor)

cell_h = h / num_rows
cell_w = w / num_cols

for r in range(num_rows):
    for c in range(num_cols):
        x0 = c * cell_w
        y0 = r * cell_h
        x1 = (c + 1) * cell_w
        y1 = (r + 1) * cell_h

        # find first keypoint whose location is inside this cell
        for kp, desc in zip(keypoints, descriptors):
            x, y = kp.pt
            if x0 <= x < x1 and y0 <= y < y1:
                label = f"Cell(r{r},c{c})"
                selected.append((label, kp, desc))
                break  # move to next cell

print(f"\nSelected {len(selected)} keypoints from different regions.")

# If too many, just keep a few to print
max_show = 5
selected = selected[:max_show]

# ============ 4. Print patterns for each selected keypoint ============
for label, kp, desc in selected:
    print_descriptor_pattern(label, kp, desc)

# ============ 5. Show Hamming distances between them ============
def hamming(d1, d2):
    return cv2.norm(d1, d2, cv2.NORM_HAMMING)

n = len(selected)
print("\nPairwise Hamming distances between selected descriptors:")
for i in range(n):
    for j in range(i + 1, n):
        di = selected[i][2]
        dj = selected[j][2]
        d = hamming(di, dj)
        print(f"{selected[i][0]} vs {selected[j][0]}: {d}")
