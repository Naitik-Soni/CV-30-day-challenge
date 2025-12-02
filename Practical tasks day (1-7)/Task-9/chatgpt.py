import cv2
import numpy as np

def detect_fire_regions(frame, min_area=1500, solidity_thresh=0.5):
    """
    Input:  BGR frame (image)
    Output: annotated_frame, fire_mask, list_of_boxes[(x, y, w, h), ...]
    """

    # 1. Smooth to reduce noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # 2. Convert to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 3. Fire-like HSV ranges (tune for your lighting/environment!)
    # Range 1: red–yellow
    lower1 = np.array([0,   80, 150])   # H, S, V
    upper1 = np.array([25, 255, 255])

    # Range 2: upper reds (HSV wrap-around)
    lower2 = np.array([160, 80, 150])
    upper2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    fire_mask = cv2.bitwise_or(mask1, mask2)

    # 4. Morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Find contours
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    annotated = frame.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # filter small blobs

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)

        # Filter very thin / extreme shapes
        if aspect < 0.2 or aspect > 5:
            continue

        # Solidity = area / convex hull area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = area / hull_area

        if solidity < solidity_thresh:
            continue

        boxes.append((x, y, w, h))

        # Draw bounding box + label
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(annotated, "FIRE", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return annotated, fire_mask, boxes


if __name__ == "__main__":
    # ---- CHANGE THIS PATH TO YOUR IMAGE ----
    img_path = r"fire.jpg"

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: could not read image at {img_path}")
        exit(1)

    annotated, fire_mask, boxes = detect_fire_regions(img)

    print("Detected fire regions (x, y, w, h):")
    for b in boxes:
        print(b)

    # Show results
    cv2.imshow("Original", img)
    cv2.imshow("Fire Mask", fire_mask)
    cv2.imshow("Fire Detection", annotated)

    # Press any key to close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optional: save outputs
    cv2.imwrite("fire_mask.png", fire_mask)
    cv2.imwrite("fire_detected.png", annotated)
