import cv2
from get_images_path import get_files_list
from perform_panorama import get_panorama

IMAGES_FOLDER_PATH = r"Input"
SIFT = cv2.SIFT_create()

def get_sift_features(image):
    return SIFT.detectAndCompute(image, None)

def resize_image(image, fc = 0.15):
    return cv2.resize(image, None, None, fx=fc, fy=fc, interpolation=cv2.INTER_AREA)

images_path = get_files_list(IMAGES_FOLDER_PATH)
images = [cv2.imread(img_path) for img_path in images_path]
images = [resize_image(img) for img in images[::-1]]

panorama_image = get_panorama(images)

if panorama_image is None:
    print("No similarity found between images for creating panorama")

cv2.imshow("Panorama", panorama_image)

# for index in range(len(images)):
#     cv2.imshow(f"Image {index+1}", images[index])

cv2.waitKey(0)
cv2.destroyAllWindows()