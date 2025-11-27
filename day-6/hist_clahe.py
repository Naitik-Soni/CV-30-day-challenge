import cv2
import matplotlib.pyplot as plt

# --- 1. Load and Preprocess Image ---
# Ensure you replace the path with your actual image file path
img_path = r"..\Images\low-contrast - 1.jpg" 
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Image not found at: {img_path}")

# Resize the image for faster processing and display
h, w = img.shape[:2]
resize_factor = 1
img = cv2.resize(img, (int(w * resize_factor), int(h * resize_factor)))

# Convert to Grayscale (required for both HE and CLAHE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 2. Global Histogram Equalization (HE) ---
eq_global = cv2.equalizeHist(gray)

# --- 3. Contrast-Limited Adaptive Histogram Equalization (CLAHE) ---

# Create a CLAHE object
# clipLimit: Threshold for contrast limiting. A typical value is 2.0 to 4.0.
# tileGridSize: Size of the grid (number of rows and columns) for dividing the image. 
#               A common size is (8, 8).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Apply the CLAHE transformation
eq_clahe = clahe.apply(gray)

# --- 4. Display Results ---
cv2.imshow("Original Grayscale", gray)
cv2.imshow("1. Global Equalization (HE)", eq_global)
cv2.imshow("2. CLAHE (Adaptive)", eq_clahe)
cv2.waitKey(0)
cv2.destroyAllWindows()