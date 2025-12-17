import cv2
import numpy as np

img = cv2.imread(r"../Images/car.jpeg", 0)
template = cv2.imread(r"../Images/logo.webp", 0)

def resize(img, resize_factor = 0.5):
    return cv2.resize(img, None, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

img = resize(img, 0.5)
template = resize(template, 0.2)

h, w = template.shape

res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.rectangle(img_color, top_left, bottom_right, (0,255,0), 2)

cv2.imshow("Match", img_color)
cv2.waitKey(0)
