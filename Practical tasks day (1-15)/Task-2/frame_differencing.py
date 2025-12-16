import cv2

# cap = cv2.VideoCapture(r"./moving_cars.mp4")
cap = cv2.VideoCapture(r"./highway.mp4")
# cap = cv2.VideoCapture(r"./thief_robbery.mp4")

sift = cv2.SIFT_create()

def resize(frame, f = 0.4):
    return cv2.resize(frame, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)

def getContours(diff):
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def getBoundingRect(img_area, contours):
    bounding_rects = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.2 * peri, True)

        x, y, w, h = cv2.boundingRect(approx)
        if cv2.contourArea(cnt) >= img_area/150:
            bounding_rects.append((x,y,w,h))

    return bounding_rects

def drawBoundingRects(img, rects):
    image = img.copy()
    for rect in rects:
        x,y,w,h = rect
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)

    return image

ret, prev = cap.read()
prev = resize(prev)
frameM1 = prev
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:
    ret, currs = cap.read()
    if not ret:
        break

    currs = resize(currs)
    curr = currs.copy()
    h, w = curr.shape[:2]

    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(curr_gray, prev_gray)

    _, motion = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)

    motion = cv2.medianBlur(motion, 5)

    contours = getContours(motion)

    bounding_rects = getBoundingRect(h*w,contours)
    rect_image = drawBoundingRects(curr, bounding_rects)

    cv2.imshow("Original", currs)
    cv2.imshow("Motion", motion)
    cv2.imshow("Gray", curr_gray)
    cv2.imshow("Bounding rects", rect_image)

    prev_gray = curr_gray
    prev = curr

    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()