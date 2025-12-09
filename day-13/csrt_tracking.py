import cv2

cap = cv2.VideoCapture(r"C:\Users\baps\Documents\Naitik Soni\ComputerVision\CV-30-day-challenge\Practical tasks day (1-7)\Task-8\circle.mp4")

# Initialize tracker
tracker = cv2.TrackerCSRT_create()

def resize_frame(frame):
    resize_factor = 0.4
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w*resize_factor), int(resize_factor*h)), cv2.INTER_AREA)

# Read first frame
ret, frame = cap.read()
frame = resize_frame(frame)

# Select ROI manually
bbox = cv2.selectROI("Frame", frame, False)
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    frame = resize_frame(frame)
    if not ret:
        break

    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, "Tracking...", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    else:
        cv2.putText(frame, "Lost track!", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        bbox = cv2.selectROI("Frame", frame, False)
        tracker.init(frame, bbox)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()