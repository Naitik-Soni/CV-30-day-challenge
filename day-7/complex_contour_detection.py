import cv2

# Load the image
img = cv2.imread(r"..\Images\road.jpeg")

if img is None:
    print("Error: Could not load image 'road.jpeg'. Check path/filename.")
else:
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Adaptive Thresholding (Better for varying lighting)
    # Using THRESH_BINARY_INV to make the dark road/line white
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5) 

    # 3. Morphological Closing (To connect broken segments of the line/road)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Find Contours
    # Use cv2.RETR_CCOMP to find all contours, then filter
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print("Length of contours:", len(contours))

    # 5. Process Contours
    output_img = img.copy()

    for cnt in contours:
        # Filter by Area: Only consider contours larger than 5000 (adjust this value)
        if cv2.contourArea(cnt) > 5000:
            
            # The rest of your shape logic is not suitable for a road line, 
            # but we can still draw the bounding box for visualization.
            
            peri = cv2.arcLength(cnt, True)
            # Use a slightly larger epsilon for polyDP, or tune as needed
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True) 

            x, y, w, h = cv2.boundingRect(approx)
            
            # Draw the contour itself, or the bounding box
            cv2.drawContours(output_img, [approx], -1, (0, 255, 0), 4)

            # Draw a box for context
            # cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            cv2.putText(output_img, f"Contour Area: {int(cv2.contourArea(cnt))}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


    # 6. Display Results
    cv2.imshow("Original", img)
    cv2.imshow("Adaptive Threshold", thresh) # Check this image to see what the contour finder sees
    cv2.imshow("Detected Contours", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()