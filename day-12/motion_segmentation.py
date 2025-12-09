# motion_segmentation.py
import cv2
import numpy as np

video_path = r"../Images/grid moving.mp4"  # or 0 for webcam
cap = cv2.VideoCapture(video_path)

ret, prev = cap.read()
if not ret:
    raise SystemExit("Cannot read video")

prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # threshold magnitude to get motion areas
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, motion_mask = cv2.threshold(mag_norm, 25, 255, cv2.THRESH_BINARY)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_DILATE, kernel, iterations=2)

    # find contours and draw bounding boxes
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = frame.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # filter small blobs (tweak as needed)
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Motion Mask", motion_mask)
    cv2.imshow("Detected Motion", out)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
