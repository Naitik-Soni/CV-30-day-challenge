import cv2

cap = cv2.VideoCapture(r"../Images/car_race.mp4")

# Initialize tracker
tracker = cv2.TrackerCSRT_create()

# Read first frame
ret, frame = cap.read()

# Select ROI manually
bbox = cv2.selectROI("Frame", frame, False)
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, "Tracking...", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    else:
        cv2.putText(frame, "Lost track!", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()