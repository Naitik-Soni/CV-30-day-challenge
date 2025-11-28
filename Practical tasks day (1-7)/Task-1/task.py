import cv2

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

    # Save the processed image
    save_image(equalized_image, "Output-1.png")
    save_image(gray, "Gray-1.png")
    save_image(clahe_image, "CLAHE-1.png")
