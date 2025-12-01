import cv2

image_path = r".\car.webp"
image = cv2.imread(image_path, 0)

# image = cv2.bilateralFilter(image, 11, 75, 75)

histequ = cv2.equalizeHist(image)
clahe_obj = cv2.createCLAHE(5, (16,16))
clahe = clahe_obj.apply(image)

_, thresh = cv2.threshold(clahe, 75, 255, cv2.THRESH_BINARY)
adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

_, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


cv2.imshow(r"Original", image)
cv2.imwrite(r".\Outputs\Hist eq.png", histequ)
cv2.imwrite(r".\Outputs\CLAHE.png", clahe)
cv2.imwrite(r".\Outputs\Normal thresh.png", thresh)
cv2.imwrite(r".\Outputs\Adaptive.png", adaptive)
cv2.imwrite(r".\Outputs\Otsu.png", otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()