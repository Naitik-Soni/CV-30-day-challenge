import cv2

img_path = 'Image-1.jpg'
img = cv2.imread(img_path)

resize_factor = 0.2
new_dimensions = (int(img.shape[1] * resize_factor), int(img.shape[0] * resize_factor))
img = cv2.resize(img, new_dimensions)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray_img = cv2.Canny(gray_img, 50, 150)

ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (255, 0, 0), 2)

print("Number of contours found:", len(contours))

for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(approx)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 72, 72), 4)


cv2.imshow('Gray Image', gray_img)
cv2.imshow('Contours', img)
cv2.imshow('Thresholded Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()