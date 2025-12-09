import cv2
import numpy as np

cap = cv2.VideoCapture(r"../Images/car_race.mp4")

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Select ROI
x,y,w,h = cv2.selectROI("Frame", old_frame, False)
mask = np.zeros_like(old_gray)
mask[y:y+h, x:x+w] = 255

p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask,
                             maxCorners=100, qualityLevel=0.3, minDistance=7)

lk_params = dict(winSize=(15,15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st==1]
    good_old = p0[st==1]

    for new in good_new:
        x2,y2 = new.ravel()
        cv2.circle(frame, (int(x2),int(y2)), 3, (0,255,0), -1)

    # Update for next frame
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

    cv2.imshow("OpticalFlow Tracking", frame)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
