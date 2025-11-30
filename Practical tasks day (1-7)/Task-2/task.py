import cv2

image_path = r".\Noisy.png"
image = cv2.imread(image_path, 0)

avg_blur = cv2.blur(image, (5,5))

gaussian = cv2.GaussianBlur(image, (5,5), 0)

median = cv2.medianBlur(image, 7)
median2 = cv2.bilateralFilter(median, 7, 75, 75)

bilateral = cv2.bilateralFilter(image, 5, 71, 71)

cv2.imshow("Original", image)
cv2.imwrite(r".\Outputs\Avg.png", avg_blur)
cv2.imwrite(r".\Outputs\Gaussian.png", gaussian)
cv2.imwrite(r".\Outputs\Median.png", median)
cv2.imwrite(r".\Outputs\Median_bilateral.png", median2)
cv2.imwrite(r".\Outputs\Bilateral.png", bilateral)

cv2.waitKey(0)
cv2.destroyAllWindows()