import cv2
import numpy as np

cap = cv2.VideoCapture(r"../Images/grid moving.mp4")

feature_params = dict(maxCorners=200,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap.read()
if not ret:
    raise RuntimeError("Couldn't read first frame from video.")

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# initial feature detection (may return None)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Helper to ensure valid p0 (np.float32, shape (-1,1,2))
def make_valid_points(pts):
    if pts is None:
        return None
    pts = np.array(pts)  # ensure ndarray
    # Remove NaNs/Infs
    if not np.isfinite(pts).all():
        pts = pts[np.isfinite(pts).all(axis=2)]
    # Convert dtype and shape
    pts = pts.astype(np.float32)
    if pts.ndim == 2 and pts.shape[1] == 2:
        pts = pts.reshape(-1, 1, 2)
    return pts if pts.size else None

p0 = make_valid_points(p0)
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If no previous points, try to detect new features
    if p0 is None:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        p0 = make_valid_points(p0)
        if p0 is None:
            # nothing to track in this frame; just show frame and continue
            cv2.imshow('Sparse Optical Flow - LK', frame)
            old_gray = frame_gray.copy()
            if cv2.waitKey(30) & 0xFF == 27:
                break
            continue

    # At this point p0 should be valid
    try:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    except cv2.error as e:
        print("OpenCV error during calcOpticalFlowPyrLK:", e)
        # force re-detection next loop
        p0 = None
        old_gray = frame_gray.copy()
        continue

    # Validate outputs
    if p1 is None or st is None:
        p0 = None
        old_gray = frame_gray.copy()
        continue

    # keep only good points
    st = st.reshape(-1)
    good_new = p1[st == 1].reshape(-1, 1, 2).astype(np.float32)
    good_old = p0[st == 1].reshape(-1, 1, 2).astype(np.float32)

    # if all points lost, re-detect next iteration
    if good_new.size == 0:
        p0 = None
        old_gray = frame_gray.copy()
        continue

    # draw
    for new, old in zip(good_new.reshape(-1,2), good_old.reshape(-1,2)):
        x1, y1 = int(new[0]), int(new[1])
        x2, y2 = int(old[0]), int(old[1])
        mask = cv2.line(mask, (x1, y1), (x2, y2), (255,0,0), 2)
        frame = cv2.circle(frame, (x1, y1), 4, (0,0,255), -1)

    img = cv2.add(frame, mask)
    cv2.imshow('Sparse Optical Flow - LK', img)

    # prepare for next frame
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
