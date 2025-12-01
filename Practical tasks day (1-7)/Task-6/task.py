import cv2
import numpy as np

image_path = 'lane-2.jpg'
image = cv2.imread(image_path)

resize_factor = 0.7
h, w = image.shape[:2]
new_dimensions = (int(w * resize_factor), int(h * resize_factor))
resized_image = cv2.resize(image, new_dimensions)

gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

CLAHE_OBJ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = CLAHE_OBJ.apply(gray_image)

ksize = 13
blurred = cv2.GaussianBlur(clahe_image, (ksize, ksize), 0)
edges = cv2.Canny(blurred, threshold1=40, threshold2=170)

lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

print(f'Detected lines: {lines}')

new_lines = []

for line in lines:
    x1, y1, x2, y2 = line[0]
    if x2 - x1 == 0:
        continue
    slope = (y2 - y1) / (x2 - x1)
    if abs(slope) < 0.5:
        continue
    new_lines.append(line)

if lines is not None:
    for line in new_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Resized Grayscale Image', gray_image)
cv2.imshow('Original Image', resized_image)
cv2.imshow('CLAHE Image', clahe_image)
cv2.imshow('Edges Detected', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()