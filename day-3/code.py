import cv2
import numpy as np
import os

# --- Config ---
img_path = r"P:\Computer vision Experiments\30-Days challenge\CV-30-day-challenge\Images\deer.jpg"
resize_factor = 0.6
kernel_size = (5, 5)
kernel_shape = cv2.MORPH_CROSS   # try: MORPH_ELLIPSE, MORPH_CROSS
iterations = 1
out_dir = r".\outputs"
os.makedirs(out_dir, exist_ok=True)

# --- Load image (grayscale) ---
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load image at: {img_path}")

# --- Resize (downsample) ---
h, w = img.shape[:2]
img = cv2.resize(img, (int(w * resize_factor), int(h * resize_factor)),
                 interpolation=cv2.INTER_AREA)

# --- Structuring element ---
kernel = cv2.getStructuringElement(kernel_shape, kernel_size)

# --- Basic morphological operations ---
dilation = cv2.dilate(img, kernel, iterations=iterations)
erosion = cv2.erode(img, kernel, iterations=iterations)

# Opening: erosion then dilation
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)

# Closing: dilation then erosion
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)

# Optional extras: morphological gradient, top-hat, black-hat
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)   # edges
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)       # small bright spots
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)   # small dark spots

# --- Save outputs ---
cv2.imwrite(os.path.join(out_dir, "original.png"), img)
cv2.imwrite(os.path.join(out_dir, "dilation.png"), dilation)
cv2.imwrite(os.path.join(out_dir, "erosion.png"), erosion)
cv2.imwrite(os.path.join(out_dir, "opened.png"), opened)
cv2.imwrite(os.path.join(out_dir, "closed.png"), closed)
cv2.imwrite(os.path.join(out_dir, "gradient.png"), gradient)
cv2.imwrite(os.path.join(out_dir, "tophat.png"), tophat)
cv2.imwrite(os.path.join(out_dir, "blackhat.png"), blackhat)

# --- Display: make a combined comparison image ---
def stack_images(img_list, cols=3, scale=1.0):
    # convert to 3-channel BGR for consistent stacking
    imgs = [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in img_list]
    # pad rows to make full grid
    rows = []
    for r in range(0, len(imgs), cols):
        row_imgs = imgs[r:r+cols]
        if len(row_imgs) < cols:
            # pad with black images
            h, w = imgs[0].shape[:2]
            for _ in range(cols - len(row_imgs)):
                row_imgs.append(np.zeros((h, w, 3), dtype=np.uint8))
        row = np.hstack(row_imgs)
        rows.append(row)
    return np.vstack(rows)

labels = ["Original", "Dilation", "Erosion",
          "Opened", "Closed", "Gradient",
          "Top-hat", "Black-hat"]
images = [img, dilation, erosion, opened, closed, gradient, tophat, blackhat]
grid = stack_images(images, cols=3)

# overlay labels (small)
font = cv2.FONT_HERSHEY_SIMPLEX
h_img, w_img = images[0].shape[:2]
pad_x = 10
pad_y = 20
for idx, label in enumerate(labels):
    row = idx // 3
    col = idx % 3
    x = col * w_img + pad_x
    y = row * h_img + pad_y
    cv2.putText(grid, label, (x, y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

# show and wait
cv2.imshow("Morphology - comparisons", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()