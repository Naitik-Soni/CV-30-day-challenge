import cv2

image = cv2.imread(r"../Images/traffic_synthetic.jpg")

image = cv2.resize(image, None, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(hsv)

cv2.imshow("BGR", image)
cv2.imshow("HSV", hsv)
cv2.imshow("H", h)
cv2.imshow("S", s)
cv2.imshow("V", v)

cv2.waitKey(0)