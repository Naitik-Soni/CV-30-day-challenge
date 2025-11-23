import cv2

image_path = r"P:\Computer vision Experiments\30-Days challenge\CV-30-day-challenge\Images\deer.jpg"
image = cv2.imread(image_path, 0)

canny = cv2.Canny(image, 180, 180)

cv2.imshow("Edges", canny)
cv2.imshow("Og", image)
cv2.waitKey(0)
cv2.destroyAllWindows()