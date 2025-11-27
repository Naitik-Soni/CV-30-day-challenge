import cv2

img_path = r"..\Images\low_contrast - 1.jpg"
image = cv2.imread(img_path)

h, w = image.shape[:2]
resize_factor = 1

image = cv2.resize(image, (int(w * resize_factor), int(h * resize_factor)))

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
lab[:,:,0] = clahe.apply(lab[:,:,0])
result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

cv2.imshow("Original", image)
cv2.imshow("CLAHE in LAB Color Space", result)
cv2.waitKey(0)
cv2.destroyAllWindows()