import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(img, mask)

def make_line_points(y1, y2, slope, intercept):
    # x = (y - intercept) / slope
    if slope == 0:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, int(y1), x2, int(y2))

def average_lane_line(lines):
    # lines: list of (slope, intercept)
    if len(lines) == 0:
        return None
    slope_mean = np.mean([l[0] for l in lines])
    int_mean = np.mean([l[1] for l in lines])
    return (slope_mean, int_mean)

# --- load image ---
image_path = 'lane-2.jpg'
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(image_path)

# optional resize
resize_factor = 0.7
h, w = img.shape[:2]
img = cv2.resize(img, (int(w * resize_factor), int(h * resize_factor)), interpolation=cv2.INTER_AREA)

# --- preprocess ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_clahe = clahe.apply(gray)
blurred = cv2.GaussianBlur(gray_clahe, (13, 13), 0)  # keep odd kernel
edges = cv2.Canny(blurred, threshold1=40, threshold2=170)

# --- region of interest ---
h, w = edges.shape
# polygon roughly covering the bottom half where lanes exist (adjust if needed)
roi_vertices = np.array([
    (int(0.1 * w), h),
    (int(0.45 * w), int(0.6 * h)),
    (int(0.55 * w), int(0.6 * h)),
    (int(0.95 * w), h)
])
masked = region_of_interest(edges, roi_vertices)

# --- Hough lines ---
lines = cv2.HoughLinesP(masked,
                        rho=1,
                        theta=np.pi/180,
                        threshold=50,
                        minLineLength=50,
                        maxLineGap=30)

# safe-check
if lines is None:
    print("No lines detected.")
    cv2.imshow('Input', img)
    cv2.imshow('Edges', edges)
    cv2.imshow('ROI Masked', masked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    raise SystemExit("No lines to process")

# --- separate into left/right by slope and compute average line (slope, intercept) ---
left_lines = []
right_lines = []
slope_threshold = 0.5  # ignore near-horizontal lines; tune as needed

for l in lines:
    x1, y1, x2, y2 = l[0]
    dx = (x2 - x1)
    if dx == 0:
        continue
    slope = (y2 - y1) / dx
    intercept = y1 - slope * x1
    if abs(slope) < slope_threshold:
        continue
    # negative slope -> left lane in typical image coords (origin at top-left)
    if slope < 0:
        left_lines.append((slope, intercept))
    else:
        right_lines.append((slope, intercept))

left_avg = average_lane_line(left_lines)
right_avg = average_lane_line(right_lines)

# define y-range for drawing extrapolated lines
y_bottom = h
y_top = int(h * 0.6)  # same as ROI top

line_img = img.copy()

if left_avg is not None:
    pts = make_line_points(y_bottom, y_top, left_avg[0], left_avg[1])
    if pts is not None:
        cv2.line(line_img, (pts[0], pts[1]), (pts[2], pts[3]), (0, 255, 0), 6)

if right_avg is not None:
    pts = make_line_points(y_bottom, y_top, right_avg[0], right_avg[1])
    if pts is not None:
        cv2.line(line_img, (pts[0], pts[1]), (pts[2], pts[3]), (0, 255, 0), 6)

# overlay ROI poly and show
overlay = img.copy()
cv2.polylines(overlay, [roi_vertices], isClosed=True, color=(255, 0, 0), thickness=2)
combined = cv2.addWeighted(overlay, 0.6, line_img, 0.4, 0)

cv2.imshow('Original Resized', img)
cv2.imshow('Edges', edges)
cv2.imshow('ROI Masked', masked)
cv2.imshow('Lane Lines', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()