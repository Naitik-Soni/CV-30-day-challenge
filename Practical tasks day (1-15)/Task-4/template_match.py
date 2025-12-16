import cv2

color_image = cv2.imread('pb1.jpg')  # Load the main image in color
grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)  # Convert the color image to grayscale
template = cv2.imread('cap.png', 0)  # Load the template image in grayscale

# Get the dimensions of the template
w, h = template.shape[::-1]

# Perform template matching using cv2.matchTemplate
result = cv2.matchTemplate(grayscale_image, template, cv2.TM_CCOEFF_NORMED)

# Find the best match location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print(min_val)
print(max_val)
print(min_loc)
print(max_loc)
# Top-left corner of the match
top_left = max_loc
# Bottom-right corner of the match
bottom_right = (top_left[0] + w, top_left[1] + h)

# Draw rectangles on both grayscale and color images
grayscale_with_bbox = grayscale_image.copy()
color_with_bbox = color_image.copy()

# Draw white bounding box on grayscale image
cv2.rectangle(grayscale_with_bbox, top_left, bottom_right, (255, 255, 255), 2)

# Draw green bounding box on color image
cv2.rectangle(color_with_bbox, top_left, bottom_right, (0, 255, 0), 2)

cv2.imshow("Matched", color_with_bbox)
cv2.waitKey(0)
cv2.destroyAllWindows()