"""
Write a Python script that:
-> Loads an image (image_path same as yesterday’s or a new one).
-> Converts it to grayscale.
-> Applies and saves results for:
    - Binary thresholding
    - Adaptive Gaussian thresholding
    - Otsu’s thresholding
-> Applies and saves results for:
    - Normal blur
    - Gaussian blur
    - Median blur
    - Bilateral filter
-> Shows all results using cv2.imshow() (or matplotlib if preferred).
"""

import cv2

output_dir = "outputs"
image_path = r"P:\Computer vision Experiments\30-Days challenge\CV-30-day-challenge\Images\nature.jpg"

image = cv2.imread(image_path)

# Original image
original = image.copy()
# Grayscale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite(f"{output_dir}/Original.png", original)
cv2.imwrite(f"{output_dir}/Gray.png", gray)

# Simple/Binary thresholding
thresh_value = 127
ret_simple, binary_img = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
# Adaptive thresholding
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY, 11, 3)
# Otsu's thresholding
ret_otsu, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, 255)

cv2.imwrite(f"{output_dir}/binary.png", binary_img)
cv2.imwrite(f"{output_dir}/adaptive.png", adaptive)
cv2.imwrite(f"{output_dir}/otsu.png", otsu)

# Normal blur
norm_blur = cv2.blur(gray, (5,5))
# Gaussian blur
gauss_blur = cv2.GaussianBlur(gray, (5,5), 0)
# Median blur
median_blur = cv2.medianBlur(gray, 5)
# Bilateral filter
bilateral = cv2.bilateralFilter(gray, 5, 75, 75)

cv2.imwrite(f"{output_dir}/norm_blur.png", norm_blur)
cv2.imwrite(f"{output_dir}/gauss_blur.png", gauss_blur)
cv2.imwrite(f"{output_dir}/median_blur.png", median_blur)
cv2.imwrite(f"{output_dir}/bilateral.png", bilateral)