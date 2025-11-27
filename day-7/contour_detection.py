import cv2

img = cv2.imread(r"..\Images\rings.webp")

h, w = img.shape[:2]
resize_factor = 1
img = cv2.resize(img, (int(w * resize_factor), int(h * resize_factor)))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.equalizeHist(gray)

blur = cv2.GaussianBlur(gray, (11, 11), 0)
ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print("Length of contours:", len(contours))

cv2.drawContours(img, contours, -1, (127, 180, 255), 3)
for cnt in contours:

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(approx)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 72, 72), 4)

    # Shape classification
    sides = len(approx)

    if sides == 3:
        shape = "Triangle"
    elif sides == 4:
        shape = "Rectangle"
    elif sides > 6:
        shape = "Circle"
    else:
        shape = f"Polygon-{sides}"

    cv2.putText(img, shape, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.imshow("Contours", img)
cv2.imshow("Gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()