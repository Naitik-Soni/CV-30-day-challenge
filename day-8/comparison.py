import cv2
import numpy as np

def process_frame(frame):
    """Return side-by-side Harris vs Shi-Tomasi visualization for a single frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_f = np.float32(gray)

    # ---------- Harris ----------
    harris = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)  # just for visualization
    harris_vis = frame.copy()

    h_thresh = 0.01 * harris.max()
    harris_vis[harris > h_thresh] = [0, 0, 255]  # red for corners

    # ---------- Shi–Tomasi ----------
    shi_vis = frame.copy()
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=5000,
        qualityLevel=0.005,
        minDistance=2,
        blockSize=3,
        useHarrisDetector=False  # False => Shi–Tomasi
    )

    if corners is not None:
        corners = corners.astype(int)  # instead of np.int0 (deprecated)
        for c in corners:
            x, y = c.ravel()
            cv2.circle(shi_vis, (x, y), 3, (0, 0, 255), -1)

    # ---------- Labels ----------
    cv2.putText(harris_vis, "Harris", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(shi_vis, "Shi-Tomasi", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # ---------- Side-by-side ----------
    combo = np.hstack([harris_vis, shi_vis])
    return combo

def compare_harris_shitomasi_video(input_path, output_path, show_preview=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: cannot open video:", input_path)
        return

    # Get video properties
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output frame will be double width (Harris | Shi-Tomasi)
    out_w, out_h = width * 2, height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 20, (out_w, out_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        combo = process_frame(frame)
        out.write(combo)

        if show_preview:
            cv2.imshow("Harris vs Shi-Tomasi", combo)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit preview
                break

    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()
    print("Saved output video to:", output_path)

if __name__ == "__main__":
    # Change these paths as needed
    input_video  = "..\Images\Soccer.mp4"                     # your video file
    output_video = "harris_vs_shitomasi_soccer.mp4"       # output comparison video

    compare_harris_shitomasi_video(input_video, output_video, show_preview=False)
