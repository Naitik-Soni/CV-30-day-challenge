import cv2
# --- Upsampling Example ---
G2 = cv2.imread(r'./Outputs/G2.png')
G1 = cv2.imread(r'./Outputs/G1.png')

if G2 is None:
    raise FileNotFoundError("G2 image not loaded. Please ensure './Outputs/G2.png' exists.")    

# Upsample back to the double size
G2_up = cv2.pyrUp(G2)

print(f"G2 size: {G2.shape}")
print(f"G2 Upsampled size: {G2_up.shape}")

G1_up = cv2.pyrUp(G1)

print(f"G1 size: {G1.shape}")
print(f"G1 Upsampled size: {G1_up.shape}")

# Display (optional - requires a GUI environment like a script or Jupyter)
cv2.imwrite(r'./Outputs/G2_up.png', G2_up)
cv2.imwrite(r'./Outputs/G1_up.png', G1_up)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Note: G2_up is *not* identical to G1. Information was lost during pyrDown.
# The size of G2_up should match the size of G1 (the next level up).
# The shapes might differ slightly if the original image dimensions weren't perfect multiples of 2.