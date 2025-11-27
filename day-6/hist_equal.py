import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"..\Images\low-contrast.jpg")

h, w = img.shape[:2]
resize_factor = 0.2
img = cv2.resize(img, (int(w * resize_factor), int(h * resize_factor)))


if img is None:
    raise FileNotFoundError("Image not found. Please check the path.")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

eq = cv2.equalizeHist(gray)

cv2.imshow("Original", gray)
cv2.imshow("Equalized", eq)
cv2.waitKey(0)
cv2.destroyAllWindows()