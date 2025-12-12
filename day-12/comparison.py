# compare_dense_sparse.py
import cv2
import numpy as np

video_path = r"../Images/Man-walking.mp4"  # or 0 for webcam
cap = cv2.VideoCapture(video_path)

# LK params
feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=7, blockSize=7)
lk_params = dict(winSize=(21,21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

ret, prev = cap.read()
if not ret:
    raise SystemExit("Cannot read video")
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

hsv = np.zeros_like(prev)
hsv[..., 1] = 255

mask_lk = np.zeros_like(prev)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Dense Farneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Sparse LK
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
    if p1 is not None:
        good_new = p1[st.flatten() == 1]
        good_old = p0[st.flatten() == 1]
        for new, old in zip(good_new, good_old):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            cv2.line(mask_lk, (a, b), (c, d), (0, 255, 0), 2)
            cv2.circle(frame, (a, b), 3, (0, 255, 0), -1)

        p0 = good_new.reshape(-1, 1, 2)
    else:
        # re-detect features if lost
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        mask_lk = np.zeros_like(prev)

    # Compose side-by-side
    left = frame  # with LK overlay lines and points
    right = flow_bgr

    # resize to equal heights if needed
    h = left.shape[0]
    right = cv2.resize(right, (left.shape[1], h))

    combined = np.hstack((left, right))
    cv2.imshow("Sparse LK (left)  |  Dense Farneback (right)", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
