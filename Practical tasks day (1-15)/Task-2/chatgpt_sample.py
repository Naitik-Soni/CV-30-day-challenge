import cv2

cap = cv2.VideoCapture(r"./moving_cars.mp4")
# cap = cv2.VideoCapture(r"./thief_robbery.mp4")

def resize(frame, f = 0.4):
    return cv2.resize(frame, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)

ret, prev = cap.read()
prev = resize(prev)
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:
    ret, curr = cap.read()
    if not ret:
        break

    curr = resize(curr)

    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(curr_gray, prev_gray)

    _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    cv2.imshow("Original", curr)
    cv2.imshow("Motion", motion)

    prev_gray = curr_gray

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
