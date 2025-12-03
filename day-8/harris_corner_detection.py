import cv2
import numpy as np

# 1. Read image and convert to gray
img = cv2.imread("../Images/balcony garden.jpg")  # put your image path here

resize_factor = 0.35
h, w = img.shape[:2]
img = cv2.resize(img, (int(resize_factor*w), int(resize_factor*h)))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Convert to float32 (Harris requirement)
gray_f = np.float32(gray)

# 3. Apply Harris Corner Detector
# Parameters:
# blockSize = neighborhood size
# ksize     = Sobel kernel size for gradient
# k         = Harris free parameter (0.04–0.06 usually)
harris_response = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)

# 4. Dilate for better visualization (optional, just to make corners fatter)
harris_dilated = cv2.dilate(harris_response, None)

# 5. Threshold to select strong corners
thresh = 0.01 * harris_dilated.max()   # 1% of max response
corners_img = img.copy()
corners_img[harris_dilated > thresh] = [255, 0, 0]  # mark corners in red

# 6. Also create a heatmap visualization of the response
harris_norm = cv2.normalize(harris_response, None, 0, 255, cv2.NORM_MINMAX)
harris_norm = np.uint8(harris_norm)
harris_color = cv2.applyColorMap(harris_norm, cv2.COLORMAP_JET)

# 7. Show results
cv2.imshow("Original", img)
cv2.imshow("Harris Response (heatmap)", harris_color)
cv2.imshow("Harris Corners", corners_img)
cv2.imshow("Harris dilated", harris_dilated)
cv2.imshow("Harris Response", harris_response)
cv2.waitKey(0)
cv2.destroyAllWindows()