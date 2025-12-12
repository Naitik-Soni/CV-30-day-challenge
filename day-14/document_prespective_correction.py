import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

WINDOW_NAME = "Select 4 corners (press w to warp)"
points = []
orig_img = None
display_img = None
scale = 1.0


# -------------------------------
# 1. Order points (TL, TR, BR, BL)
# -------------------------------
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(4)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")


# -------------------------------
# 2. Perspective warp
# -------------------------------
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# -------------------------------
# 3. Redraw points + lines
# -------------------------------
def redraw():
    global display_img
    canvas = display_img.copy()

    for i, p in enumerate(points):
        cv2.circle(canvas, (int(p[0]), int(p[1])), 6, (0, 255, 0), -1)
        cv2.putText(canvas, str(i + 1), (int(p[0]) + 10, int(p[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if len(points) >= 2:
        cv2.polylines(canvas, [np.array(points, np.int32)], 
                      isClosed=(len(points) == 4), 
                      color=(255, 0, 0), thickness=2)

    cv2.putText(canvas, "Left-click:add | Right-click:undo | r:reset | w:warp | s:save | q:quit",
                (10, canvas.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow(WINDOW_NAME, canvas)


# -------------------------------
# 4. Mouse callback
# -------------------------------
def mouse_cb(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            redraw()

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
            redraw()


# -------------------------------
# 5. Main
# -------------------------------
def main():
    global orig_img, display_img, scale, points

    # --------------------
    # Select image (GUI)
    # --------------------
    root = tk.Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(
        title="Select Document Image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not img_path:
        print("No file selected.")
        return

    orig_img = cv2.imread(img_path)

    h, w = orig_img.shape[:2]
    max_disp = 1000
    scale = 1.0

    if max(h, w) > max_disp:
        scale = max_disp / max(h, w)
        display_img = cv2.resize(orig_img, (int(w * scale), int(h * scale)))
    else:
        display_img = orig_img.copy()

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_cb)
    redraw()

    warped = None

    # --------------------
    # Keyboard controls
    # --------------------
    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('r'):
            points = []
            redraw()

        elif key == ord('w'):
            if len(points) != 4:
                print("Select 4 points first.")
                continue

            pts_orig = [(p[0] / scale, p[1] / scale) for p in points]
            warped = four_point_transform(orig_img, pts_orig)
            cv2.imshow("Warped Result", warped)

        elif key == ord('s'):
            if warped is not None:
                cv2.imwrite("warped_output.jpg", warped)
                print("Saved as warped_output.jpg")

        elif key in (ord('q'), 27):  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
