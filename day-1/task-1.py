import cv2
from pathlib import Path

# === Config ===
image_path = Path(r"P:\Computer vision Experiments\30-Days challenge\CV-30-day-challenge\Images\image5.jpg")
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

resize_factor = 0.5    # scale factor for resizing (50%)

# === Load safely ===
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Could not read image at: {image_path}")

# Keep a copy of original (before any resizing)
original = image.copy()

# === Resize once (controlled) ===
h0, w0 = original.shape[:2]
new_w, new_h = int(w0 * resize_factor), int(h0 * resize_factor)
resized = cv2.resize(original, (new_w, new_h), interpolation=cv2.INTER_AREA)

# === Color conversions ===
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)           # grayscale
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)            # HSV space (for processing)

# If you want to *save a viewable* HSV image, convert back to BGR
hsv_bgr_for_saving = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# === Rotation & flipping ===
rotated_90_cw = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)  # 90° clockwise
# For arbitrary angle rotation use getRotationMatrix2D + warpAffine

flipped_horizontal = cv2.flip(resized, 1)  # 1 = horizontal, 0 = vertical, -1 = both

# === Save outputs ===
cv2.imwrite(str(out_dir / "Original.png"), original)                 # original (no resize)
cv2.imwrite(str(out_dir / "Resized.png"), resized)                   # resized
cv2.imwrite(str(out_dir / "Gray.png"), gray)                         # grayscale
cv2.imwrite(str(out_dir / "HSV_viewable.png"), hsv_bgr_for_saving)   # HSV converted back for viewing
cv2.imwrite(str(out_dir / "Rotated_90CW.png"), rotated_90_cw)
cv2.imwrite(str(out_dir / "Flipped_Horizontal.png"), flipped_horizontal)

print("Saved images to", out_dir.resolve())