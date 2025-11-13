import cv2
import numpy as np

image_path = r"P:\Computer vision Experiments\30-Days challenge\CV-30-day-challenge\Images\full moon.jpg"
og = cv2.imread(image_path)

image = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)

ret, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((5,5), dtype=np.uint8)

eroded = cv2.erode(thresholded, kernel, 9)
diluted = cv2.dilate(eroded, kernel, 11)

mask = cv2.bitwise_not(diluted)

final_image = cv2.bitwise_and(og, og, mask=mask, )

lower_black = np.array([0, 0, 0])
upper_black = np.array([0, 0, 0])
black_mask = cv2.inRange(final_image, lower_black, upper_black)

b,g,r = cv2.split(final_image)

alpha = np.ones(b.shape, dtype=b.dtype) * 255
alpha[black_mask == 255] = 0
alpha[black_mask != 255] = 255

new_final_image = cv2.merge((b,g,r, alpha))


# cv2.imshow("Dilated", diluted)
# cv2.imshow("Eroded", eroded)
# cv2.imshow("Binary", thresholded)
# cv2.imshow("Image", image)

cv2.imwrite("outputs/final.png", new_final_image)

cv2.waitKey(0)
cv2.destroyAllWindows()