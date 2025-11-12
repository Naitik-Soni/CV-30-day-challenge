import cv2
import numpy as np
from matplotlib import pyplot as plt

# --- 1. Load the Image ---
# NOTE: Update the path as needed. The synthetic image creation 
# simulates uneven lighting for better demonstration of adaptive methods.
try:
    img = cv2.imread(r'P:\Computer vision Experiments\30-Days challenge\CV-30-day-challenge\Images\Autumn.jpg', 0)
    if img is None:
        raise FileNotFoundError("Image file not found at the specified path.")
except FileNotFoundError as e:
    # Creating a synthetic image with a gradient for demonstration if a file isn't found.
    print("⚠️ Using a synthetic image as the specified file was not found.")
    rows, cols = 200, 200
    synthetic_img = np.zeros((rows, cols), dtype=np.uint8)
    
    # Create a gradient from dark (top-left) to light (bottom-right)
    for i in range(rows):
        for j in range(cols):
            synthetic_img[i, j] = int(255 * (i + j) / (rows + cols))
    
    # Add a dark square object in the center (the 'foreground')
    synthetic_img[70:130, 70:130] = np.clip(synthetic_img[70:130, 70:130] - 50, 0, 255)
    img = synthetic_img


# --- Global Thresholding Parameters ---
MAX_VAL = 255 
T_SIMPLE = 127 # Manual threshold for Simple method

# --- Adaptive Thresholding Parameters ---
BLOCK_SIZE = 11  # Must be an odd number (size of the local neighborhood)
C = 2            # Constant subtracted from the mean/weighted mean


# ------------------------------------------
# --- 2. Simple (Global) Thresholding ---
# ------------------------------------------
ret_simple, thresh_simple = cv2.threshold(
    img, 
    T_SIMPLE, 
    MAX_VAL, 
    cv2.THRESH_BINARY
)

# ------------------------------------------
# --- 3. Otsu's Thresholding ---
# ------------------------------------------
T_otsu, thresh_otsu = cv2.threshold(
    img, 
    0, 
    MAX_VAL, 
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# ------------------------------------------
# --- 4. Adaptive Mean Thresholding ---
# ------------------------------------------
thresh_mean = cv2.adaptiveThreshold(
    img, 
    MAX_VAL, 
    cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, 
    BLOCK_SIZE, 
    C
)

# ------------------------------------------
# --- 5. Adaptive Gaussian Thresholding ---
# ------------------------------------------
thresh_gaussian = cv2.adaptiveThreshold(
    img, 
    MAX_VAL, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 
    BLOCK_SIZE, 
    C
)


# ------------------------------------------
# --- 6. Display All Results ---
# ------------------------------------------
images = [img, thresh_simple, thresh_otsu, thresh_mean, thresh_gaussian]
titles = [
    'Original Grayscale Image', 
    f'Simple (T={T_SIMPLE})', 
    f"Otsu's (T={T_otsu:.2f})",
    f'Adaptive Mean (B={BLOCK_SIZE}, C={C})', 
    f'Adaptive Gaussian (B={BLOCK_SIZE}, C={C})'
]

# Plotting all 5 images for side-by-side comparison
plt.figure(figsize=(20, 8)) # Increased figure size for 5 plots
for i in range(len(images)):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=10)
    plt.xticks([]), plt.yticks([])

plt.show()