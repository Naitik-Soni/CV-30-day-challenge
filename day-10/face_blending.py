import cv2
import numpy as np

image1 = cv2.imread(r"../Images/Face.jpg")
image2 = cv2.imread(r"../Images/city.jpg")

image1 = cv2.resize(image1, (600, 600), cv2.INTER_AREA)
image2 = cv2.resize(image2, (600, 600), cv2.INTER_AREA)

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

alpha = 0.5
# image1[gray1>250] = [0,255,0]
blended_image = cv2.addWeighted(image1, alpha, image2, 1-alpha, 0)
selected_blended_pixels = blended_image[gray1 < 251]
image1[gray1 < 251] = selected_blended_pixels

cv2.imshow("Image 1", image1)
cv2.imshow("Image 2", image2)
cv2.imshow("Gray 1", gray1)
cv2.imshow("Gray 2", gray2)

cv2.waitKey(0)
cv2.destroyAllWindows()
