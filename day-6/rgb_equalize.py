import cv2

img_path = r"..\Images\low_contrast - 1.jpg"
image = cv2.imread(img_path)

h, w = image.shape[:2]
resize_factor = 1

image = cv2.resize(image, (int(w * resize_factor), int(h * resize_factor)))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
b,g,r = cv2.split(image)

eq_r = cv2.equalizeHist(r)
eq_g = cv2.equalizeHist(g)
eq_b = cv2.equalizeHist(b)

eq_image = cv2.merge((eq_b, eq_g, eq_r))
cv2.imshow("Original", image)
cv2.imshow("Equalized RGB Image", eq_image)
cv2.waitKey(0)
cv2.destroyAllWindows()