import cv2
from number_recognizer import recognize_number

imgpath = 'sudoku.webp'
img = cv2.imread(imgpath)

resize_factor = 0.15
h, w = img.shape[:2]
new_size = (int(w * resize_factor), int(h * resize_factor))
resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

thresholded = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)[1]

contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(resized_img, contours, -1, (0, 255, 0), 2)

bounding_rect = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    # print("Area:", area, "Vertices:", len(approx))

    if area < 2500 and len(approx) == 4 and area  > 500:
        bounding_rect.append(cv2.boundingRect(approx))
        cv2.drawContours(resized_img, [cnt], -1, (0, 255, 0), 2)

for rect in bounding_rect:
    x, y, w, h = rect
    check_sum = 5
    roi = thresholded[y+check_sum:y+h-check_sum, x+check_sum:x+w-check_sum]
    num_meta = recognize_number(roi)
        

print(f'Number of detected squares with area < 2000: {bounding_rect}')

cv2.imshow('Resized Grayscale Image', gray)
cv2.imshow('Original Image', resized_img)
cv2.imshow('Thresholded Image', thresholded)
cv2.waitKey(0)