import cv2
import numpy as np
import os
import re

sift = cv2.ORB_create(2000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def load_image(image_path, resize = False, resize_factor = 1.0):
    image = cv2.imread(image_path)
    if resize:
        return resize_image(image, resize_factor)
    return image

def resize_image(image, resize_factor = 1.0):
    h, w = image.shape[:2]
    new_h, new_w = int(h*resize_factor), int(w*resize_factor)

    return cv2.resize(image, (new_w, new_h), cv2.INTER_AREA)

def get_features(image):
    return sift.detectAndCompute(image, None)

def get_good_matches(descriptor1, descriptor2, top_matches=100):
    matches = matcher.match(descriptor1, descriptor2)
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches_len = min(100, len(matches))

    return matches[:good_matches_len]

def get_final_points(keypoints1, keypoints2, good_matches):
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    return src_pts, dst_pts

def find_homography(src, dst):
    return cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

def get_transformed_image(image2, homography_matrix, w):
    return cv2.warpPerspective(image2, homography_matrix, (w, image2.shape[0]))

def natural_key(text):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', text)]

def get_files_list(folder_path, ext=".jpg"):
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(ext)
    ]

    files_sorted = sorted(files, key=lambda x: natural_key(os.path.basename(x)))

    return files_sorted[::-1]

def show_image(image, title="Default title"):
    cv2.imshow(title, image)