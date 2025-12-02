import cv2
import time

cap = cv2.VideoCapture("circle.mp4")

fps_list = []
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    fps_list.append(fps)
    if len(fps_list) > 10:  # smooth average over 10 frames
        fps_list.pop(0)

    avg_fps = sum(fps_list) / len(fps_list)

    cv2.putText(frame, f"FPS: {avg_fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("FPS Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
