import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load an image from the specified file path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image

def resize_image(image, scale):
    """Resize the image to the specified width and height."""
    height = int(image.shape[0] * scale)
    width = int(image.shape[1] * scale)

    return cv2.resize(image, (width, height))

def save_image(image, output_path):
    """Save the image to the specified file path."""
    cv2.imwrite(output_path, image)

def equalize_image(image):
    """Apply histogram equalization to the image."""
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3:  # Color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return gray, equalized
    else:
        raise ValueError("Unsupported image format")
    
def image_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 2:  # Grayscale image
        return clahe.apply(image)
    elif len(image.shape) == 3:  # Color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe_image = clahe.apply(gray)
        return clahe_image
    else:
        raise ValueError("Unsupported image format")
    
def get_histogram_image(gray_img):
    """
    Takes a grayscale image (2D numpy array) and returns the histogram plot as an RGB image.
    Works on TkAgg, Agg, Qt5Agg backends.
    """

    if len(gray_img.shape) != 2:
        raise ValueError("Input must be a grayscale (2D) image")

    # Create matplotlib figure
    fig = plt.figure(figsize=(4, 3), dpi=100)
    plt.hist(gray_img.ravel(), bins=256, range=(0, 256))
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()

    fig.canvas.draw()

    # Try RGB first
    try:
        data = fig.canvas.tostring_rgb()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)

    except AttributeError:
        # Fallback for TkAgg → ARGB → convert to RGB
        data = fig.canvas.tostring_argb()
        w, h = fig.canvas.get_width_height()
        argb = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
        rgb = argb[:, :, 1:4]  # drop alpha
        img = rgb.copy()

    plt.close(fig)
    return img

if __name__ == "__main__":
    input_path = "Image-1.webp"
    scale = 0.5

    # Load the image
    image = load_image(input_path)

    # Resize the image
    resized_image = resize_image(image, scale)

    # Equalize the image
    gray, equalized_image = equalize_image(resized_image)
    clahe_image = image_clahe(resized_image)
    # Get histogram plot of the grayscale image
    histogram_image = get_histogram_image(gray)
    # Get histogram plot of the equalized image
    histogram_equalized_image = get_histogram_image(equalized_image)
    # Get histogram plot of the CLAHE image
    histogram_clahe_image = get_histogram_image(clahe_image)

    # Save the processed image
    save_image(equalized_image, "Outputs/Equalized-1.png")
    save_image(gray, "Outputs/Gray-1.png")
    save_image(clahe_image, "Outputs/CLAHE-1.png")

    save_image(histogram_image, "Outputs/Histogram-1.png")
    save_image(histogram_equalized_image, "Outputs/Histogram-Equalized-1.png")
    save_image(histogram_clahe_image, "Outputs/Histogram-CLAHE-1.png")
