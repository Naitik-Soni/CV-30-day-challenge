import cv2
import numpy as np

image_path = r".\input1.jpg"
img = cv2.imread(image_path)

resize_f = 0.9
img = cv2.resize(img, None, None, fx = resize_f, fy = resize_f, interpolation=cv2.INTER_LINEAR)

def get_img_area(img):
    h, w = img.shape[:2]

    return int(h*w)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(gray, 90, 150)

contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(img, contours, -1, (255, 0, 0), 2)

source_points = []

for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(approx)

    sides = len(approx)
    area = cv2.contourArea(cnt)

    if sides == 4 and area >= get_img_area(img)/10:
        points = [arr[0].tolist() for arr in approx]
        source_points.extend(points)
        break

def order_points_min(pts):
    pts = np.array(pts)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

h, w = img.shape[:2]
destiny_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
source_points = order_points_min(source_points)

M = cv2.getPerspectiveTransform(np.float32(source_points), destiny_points)
warped = cv2.warpPerspective(img, M, (w, h))

cv2.imshow("Original", img)
cv2.imshow("Warped", warped)
# cv2.imwrite(r"./Outputs/auto_warped.jpg", warped)

cv2.waitKey(0)
cv2.destroyAllWindows()