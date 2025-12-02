import cv2

def recognize_number(image):
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours) == 0:
        return None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        print(f'Area: {area}, Perimeter: {peri}, Vertices: {len(approx)}')

        num = predict_number(int(area), int(peri), int(len(approx)))
        print("Predicted Number:", num)

        cv2.imshow('Contour', image)
        cv2.waitKey(0)

        return area, peri, len(approx)
    
    return None

def predict_number(area, perimeter, vertices):
    if area < 35 and perimeter < 60:
        return 1
    elif area in range(60, 101) and perimeter in range(90, 101) and (vertices == 9 or vertices == 10):
        return 2
    elif area in range(60, 101) and perimeter in range(90, 101) and (vertices > 10):
        return 3
    elif area in range(100, 151) and perimeter in range(50, 71) and vertices in (9, 10, 11):
        return 4
    elif area in range(75, 95) and perimeter in range(100, 111) and vertices in (12, 13, 14):
        return 5
    elif area in range(110, 161) and perimeter in range(70, 91) and vertices >= 11:
        return 6
    elif area in range(25, 60) and perimeter in range(60, 81) and vertices in (5, 6, 7, 8, 9):
        return 7
    elif area > 180 and perimeter > 55 and vertices >= 10:
        return 8
    elif area > 130 and perimeter > 75 and vertices > 10:
        return 9
    else:
        return None