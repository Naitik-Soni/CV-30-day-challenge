import cv2
import numpy as np
import math

img_path = r"P:\Computer vision Experiments\30-Days challenge\CV-30-day-challenge\Images\road.jpeg"  # your file

# ---------- TUNABLE PARAMETERS ----------
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

CLAHE_CLIP = 2.0
CLAHE_TILE = (8,8)

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

# Canny auto thresholds factor (median-based)
CANNY_SIGMA = 0.33

# HoughLinesP params
HOUGH_RHO = 1
HOUGH_THETA = np.pi/180
HOUGH_THRESHOLD = 80          # accumulator threshold — increase to be stricter
MIN_LINE_LENGTH = 80          # discard short lines (increase to remove noisy short segments)
MAX_LINE_GAP = 20

# Angle filtering: keep lines whose angle is within ANGLE_TOLERANCE of vertical (in degrees).
# For your image the road center stripe is roughly vertical; adjust as needed.
TARGET_ANGLE_DEG = 90.0
ANGLE_TOLERANCE = 20.0

# ROI fraction (y_start..y_end as a fraction of image height). Keep lower central region.
ROI_Y_START = 0.45  # from top (0) to bottom (1). 0.45 means start at 45% height (upper bound)
ROI_Y_END = 1.0

# ---------- Helper functions ----------
def auto_canny_thresholds(gray, sigma=CANNY_SIGMA):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return lower, upper

def angle_of_line(x1,y1,x2,y2):
    # angle in degrees: 0 = horizontal right, 90 = vertical upward
    dx = x2 - x1
    dy = y1 - y2   # invert because image y grows downward
    ang = math.degrees(math.atan2(dy, dx))
    # normalize to [0,180)
    if ang < 0:
        ang += 180
    return ang

# ---------- Pipeline ----------
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(img_path + " not found")

h, w = img.shape[:2]

# 1) Denoise while preserving edges
den = cv2.bilateralFilter(img, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
# alternative: cv2.fastNlMeansDenoisingColored if you want stronger denoising

# 2) Convert to grayscale and enhance local contrast (CLAHE)
gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
gray_clahe = clahe.apply(gray)

# 3) Morphological open to remove small texture
morph = cv2.morphologyEx(gray_clahe, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)

# 4) Canny with automatic thresholds
low, high = auto_canny_thresholds(morph, sigma=CANNY_SIGMA)
edges = cv2.Canny(morph, low, high, apertureSize=3)

# 5) Mask ROI (keep lower center area)
mask = np.zeros_like(edges)
y1 = int(ROI_Y_START * h)
y2 = int(ROI_Y_END * h)
x1 = int(0.05 * w)
x2 = int(0.95 * w)
mask[y1:y2, x1:x2] = 255
edges_roi = cv2.bitwise_and(edges, mask)

# optional further morphological closing to connect broken edges
edges_clean = cv2.morphologyEx(edges_roi, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=1)

# 6) Probabilistic Hough
lines = cv2.HoughLinesP(edges_clean, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                        minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)

# Prepare an output image for visualization
out = img.copy()
vis = cv2.cvtColor(edges_clean, cv2.COLOR_GRAY2BGR)

filtered_lines = []
if lines is not None:
    for l in lines:
        x1,y1,x2,y2 = l[0]
        ang = angle_of_line(x1,y1,x2,y2)
        # compute difference to target angle (circular difference)
        diff = abs((ang - TARGET_ANGLE_DEG + 90) % 180 - 90)
        if diff <= ANGLE_TOLERANCE:
            filtered_lines.append((x1,y1,x2,y2,ang))
            cv2.line(out, (x1,y1),(x2,y2),(0,0,255),2)  # red lines (kept)
        else:
            # draw rejected lines faintly if you want to debug
            cv2.line(vis, (x1,y1),(x2,y2),(100,100,100),1)

# 7) (Optional) Merge/cluster nearly-collinear/nearby lines to reduce duplicates
# Simple approach: for each line, extend and average similar ones.
def merge_lines(lines, distance_thresh=20, angle_thresh=5):
    merged = []
    used = [False]*len(lines)
    for i, (x1,y1,x2,y2,ang) in enumerate(lines):
        if used[i]:
            continue
        group = [(x1,y1,x2,y2,ang)]
        used[i] = True
        for j, (xx1,yy1,xx2,yy2,ang2) in enumerate(lines):
            if used[j]: continue
            if abs(ang - ang2) <= angle_thresh:
                # distance between midpoints
                mid_i = ((x1+x2)/2, (y1+y2)/2)
                mid_j = ((xx1+xx2)/2, (yy1+yy2)/2)
                dist = math.hypot(mid_i[0]-mid_j[0], mid_i[1]-mid_j[1])
                if dist <= distance_thresh:
                    group.append((xx1,yy1,xx2,yy2,ang2))
                    used[j] = True
        # average group into one long line (approx)
        xs = [a for g in group for a in (g[0], g[2])]
        ys = [b for g in group for b in (g[1], g[3])]
        merged.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), sum(g[4] for g in group)/len(group)))
    return merged

merged = merge_lines(filtered_lines, distance_thresh=30, angle_thresh=6)
out2 = img.copy()
for (x1,y1,x2,y2,ang) in merged:
    cv2.line(out2, (x1,y1),(x2,y2),(0,0,255),3)

# Save results for inspection
cv2.imwrite("outputs/hough_edges.png", vis)
cv2.imwrite("outputs/hough_filtered.png", out)
cv2.imwrite("outputs/hough_merged.png", out2)

print("saved: /outputs/hough_edges.png (rejected lines), /outputs/hough_filtered.png (kept lines), /outputs/hough_merged.png (merged)")