import cv2
import numpy as np

# Load the image (use a simple filename for example)
# Replace 'input_image.jpg' with your actual image file
try:
    img = cv2.imread(r'../Images/deer.jpg')
    if img is None:
        raise FileNotFoundError("Image not loaded.")

    # --- Gaussian Pyramid Example ---
    
    # Check if image is large enough for downsampling
    if img.shape[0] < 2 or img.shape[1] < 2:
        print("Image is too small for downsampling.")
    else:
        # Create the first level of the Gaussian Pyramid (G1)
        G1 = cv2.pyrDown(img)
        # Create the second level (G2)
        G2 = cv2.pyrDown(G1)

        print(f"Original Image size: {img.shape}")
        print(f"Gaussian Level 1 size: {G1.shape}")
        print(f"Gaussian Level 2 size: {G2.shape}")

        # Display (optional - requires a GUI environment like a script or Jupyter)
        cv2.imwrite(r'./Outputs/Original.png', img)
        cv2.imwrite(r'./Outputs/G1.png', G1)
        cv2.imwrite(r'./Outputs/G2.png', G2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure 'input_image.jpg' exists.")
except Exception as e:
    print(f"An error occurred: {e}")