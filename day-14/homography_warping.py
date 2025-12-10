import cv2
import numpy as np

img2 = cv2.imread(r"../Images/atom_og.jpg")
img1 = cv2.imread(r"../Images/atom_ro.webp")

orb = cv2.ORB_create(2000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches
matches = sorted(matches, key=lambda x: x.distance)
good = matches[:100]

src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.25)

result = cv2.warpPerspective(img1, H, 
            (img2.shape[1], img2.shape[0]))


cv2.imshow("Stitched", result)
cv2.imshow("Image1", img1)
cv2.imshow("Image2", img2)
cv2.waitKey(0)
