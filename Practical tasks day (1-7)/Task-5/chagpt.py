import cv2
import numpy as np

image_path = 'coins.jpg'   # developer note: file is available here

# --- load & optional resize ---
image = cv2.imread(image_path)
if image is None:
    raise RuntimeError("Image not found at: " + image_path)

# optional: downscale a bit if very large (keep aspect)
resize_factor = 0.9
h, w = image.shape[:2]
new_dimensions = (int(w * resize_factor), int(h * resize_factor))
image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

# --- preprocessing for HoughCircles (use blurred grayscale, not Canny) ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Smooth: Gaussian is a good default for circles. You can also try cv2.medianBlur.
gray_blur = cv2.medianBlur(gray, 5)

# (Optional) Try a bilateral filter if you want to keep edges but remove texture noise
# gray_blur = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

# --- HoughCircles on the blurred grayscale image ---
# Notes:
# - dp: inverse ratio of accumulator resolution to image resolution
# - minDist: min distance between circle centers (pixels)
# - param1: higher threshold for internal Canny (we do NOT pass a Canny image)
# - param2: accumulator threshold for center detection (lower -> more false circles)
# - minRadius, maxRadius: limits for circle sizes (tune for your coins)
circles = cv2.HoughCircles(
    gray_blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=40,
    param1=120,   # Canny high threshold (internal). 100-200 is typical.
    param2=40,    # Accumulator threshold: lower -> more circles. Try 25-50.
    minRadius=18, # adjust to coins in your resized image
    maxRadius=80
)

output = image.copy()
if circles is not None:
    # convert to integer pixel coordinates
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)   # circle outline
        cv2.circle(output, (x, y), 2, (255, 0, 0), 5)   # center dot
        print("Detected circle - center: ({}, {}), radius: {}".format(x, y, r))
else:
    print("No circles detected. Try adjusting param1/param2/minRadius/maxRadius or preprocessing.")

# show images
cv2.imshow('Gray (equalized + blurred)', gray_blur)
cv2.imshow('Detected Circles', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save output for convenience
cv2.imwrite('Outputs/coins_detected.jpg', output)
print("Outputs/coins_detected.jpg")
