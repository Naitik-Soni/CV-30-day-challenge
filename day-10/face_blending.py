import cv2
import numpy as np

def load_resize_images():
    face = cv2.imread(r"../Images/Face.jpg")
    city = cv2.imread(r"../Images/forest.jpgr")

    face = cv2.resize(face, (400, 400), cv2.INTER_AREA)
    city = cv2.resize(city, (400, 400), cv2.INTER_AREA)
    return face, city

def get_face_mask(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    # Detect face region: white bg threshold
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Smooth edge so blending looks soft
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask = cv2.dilate(mask, kernel, iterations=5)
    mask = cv2.erode(mask, kernel, iterations=5)

    mask_f = mask.astype(np.float32) / 255.0
    return mask, mask_f

def alpha_blend_face_only(face, city, mask_f, alpha=0.7):
    """
    α applies to only face areas.
    outside mask -> pure city
    inside mask -> blended city + face
    """

    # expand for 3 channels
    mask_f = mask_f[:, :, None]

    # blending inside face area only
    blended = city * (1 - mask_f) + (face * alpha + city * (1 - alpha)) * mask_f

    return blended.astype(np.uint8)

def perform_or_operation(result, mask):
    new_mask = cv2.bitwise_not(mask)
    merged_mask = cv2.merge((new_mask, new_mask, new_mask))

    return cv2.bitwise_or(result, merged_mask)

def main():
    face, city = load_resize_images()

    mask, mask_f = get_face_mask(face)

    result = alpha_blend_face_only(face, city, mask_f, alpha=0.75)

    final_result = perform_or_operation(result, mask)

    cv2.imshow("Face", face)
    cv2.imshow("City", city)
    cv2.imshow("Mask (face region)", mask)
    cv2.imshow("Blended Result (Only Face)", result)
    cv2.imshow("Final blended result", final_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
