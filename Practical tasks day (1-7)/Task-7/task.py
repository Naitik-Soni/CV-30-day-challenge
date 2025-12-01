import cv2

image_path = "car-1.jpg"
image = cv2.imread(image_path)

resize_factor = 0.8
h, w = image.shape[:2]
new_dimensions = (int(w * resize_factor), int(h * resize_factor))
resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

equalized = cv2.equalizeHist(gray)

thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(resized_image, contours, -1, (0, 255, 0), 2)

number_plate = None

for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)

    image_area = resized_image.shape[0] * resized_image.shape[1]

    # if len(approx) == 4:
    if len(approx) == 4 and area >= image_area / 30:
        print(area, resized_image.shape[0]*resized_image.shape[1])
        number_plate = resized_image[y:y + h, x:x + w]
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(resized_image, "Object", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Resized Grayscale Image", gray)
cv2.imshow("Original Image", resized_image)
cv2.imshow("Threshed Edges with Contours", thresh)
cv2.imshow("Equalized Image", equalized)
if number_plate is not None:
    cv2.imshow("Number Plate", number_plate)
    cv2.imwrite("Outputs/Number plate.png", number_plate)
cv2.waitKey(0)
cv2.destroyAllWindows()