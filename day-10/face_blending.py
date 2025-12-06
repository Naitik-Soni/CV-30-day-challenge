import cv2
import numpy as np

def load_resize_images():
    image1 = cv2.imread(r"../Images/Face.jpg")
    image2 = cv2.imread(r"../Images/city.jpg")

    image1 = cv2.resize(image1, (600, 600), cv2.INTER_AREA)
    image2 = cv2.resize(image2, (600, 600), cv2.INTER_AREA)

    return image1, image2

def return_gray_images(images):
    gray1 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)

    return gray1, gray2

def simple_blend(images, grays):
    alpha = 0.5
    # image1[gray1>250] = [0,255,0]
    blended_image = cv2.addWeighted(images[0], alpha, images[1], 1-alpha, 0)
    selected_blended_pixels = blended_image[grays[0] < 251]
    images[0][grays[0] < 251] = selected_blended_pixels

    return images

def blur_image(image):
    return cv2.GaussianBlur(image, (5,5), 0)

def main():
    image1, image2 = load_resize_images()
    gray1, gray2 = return_gray_images((image1, image2))

    blurred = blur_image(gray1)
    image1, image2 = simple_blend((image1, image2), (blurred, gray2))


    cv2.imshow("Image 1", image1)
    cv2.imshow("Image 2", image2)
    cv2.imshow("Gray 1", gray1)
    cv2.imshow("Blurred", blurred)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()