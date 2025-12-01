import cv2

image_path = 'coins.jpg'

image = cv2.imread(image_path)

resize_factor = 0.9
h, w = image.shape[:2]
new_dimensions = (int(w * resize_factor), int(h * resize_factor))
image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
og_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.medianBlur(og_gray, 5)
bw = cv2.Canny(gray, 220, 200)

circles = cv2.HoughCircles(bw, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=50, param2=30, minRadius=20, maxRadius=50)

if circles is not None:
    circles = cv2.convertScaleAbs(circles)

    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
        print("Radius:", i[2])

cv2.imshow('Grayscale Image', gray)
cv2.imshow('Original Image', og_gray)
cv2.imshow('Canny', bw)
cv2.imshow('Detected Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()