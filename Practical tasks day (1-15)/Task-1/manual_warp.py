import cv2
import numpy as np

image_path = r".\input1.jpg"
img = cv2.imread(image_path)

f = 0.8
img = cv2.resize(img, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
image = img.copy()

if image is None:
    print("ERROR: Image not loaded")
    exit()

def hw_selected_points(points):
    p1, p2, p3, p4 = points

    point1 = np.array(p1)
    point2 = np.array(p2)
    point3 = np.array(p3)
    point4 = np.array(p4)

    w1 = cv2.norm(point1 - point2, cv2.NORM_L2)
    w2 = cv2.norm(point3 - point4, cv2.NORM_L2)
    h1 = cv2.norm(point1 - point4, cv2.NORM_L2)
    h2 = cv2.norm(point3 - point2, cv2.NORM_L2)

    return max(h1, h2), max(w1, w2)

def select_points(image):
    points = []

    def mouse_cb(event, x, y, flags, param):
        global image
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Image", image)

    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", mouse_cb)

    while True:
        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

        if len(points) == 4:
            break

    return points

points = select_points(image)
print("Selected points", points)

(h, w) = image.shape[:2]

target_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
source_points = np.float32(points)

M = cv2.getPerspectiveTransform(source_points, target_points)
warped = cv2.warpPerspective(img, M, (w, h))

cv2.imshow("Warped", warped)
cv2.imwrite(r"./Outputs/warped_manually.jpg", warped)

cv2.waitKey(0)
cv2.destroyAllWindows()