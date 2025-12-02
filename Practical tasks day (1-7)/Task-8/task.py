import cv2
import numpy as np

video_path = "circle.mp4"
cap = cv2.VideoCapture(video_path)

center = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w  = frame.shape[:2]
    resize_Factor = 0.5
    frame = cv2.resize(frame, (int(w * resize_Factor), int(h * resize_Factor)))
    frame = frame[210:740, :]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    black = np.zeros_like(frame, dtype=np.uint8)

    # edges = cv2.Canny(blurred, 50, 150)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=50, param2=30, minRadius=1, maxRadius=50)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center.append((i[0], i[1]))
            cv2.circle(black, (i[0], i[1]), i[2], (255, 255, 255), -1)

        l = len(center)
        for j in range(1, len(center)):
            cv2.line(black, center[j-1], center[j], (255, 255, 255), 2)

    cv2.imshow("Video Frame", frame)
    cv2.imshow("Tracking", black)
    # cv2.imshow("Edges", edges)    

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break