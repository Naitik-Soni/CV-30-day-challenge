from panorama_utils import *

resize_factor = 0.2

def get_homography_image(prev_img, img2, final_width):

    # Read the 2nd image
    image2 = load_image(img2, True, resize_factor)

    # Get features of both image with ORB
    kp1, dst1 = get_features(prev_img)
    kp2, dst2 = get_features(image2)

    # Find good matches with BFmatcher
    good_matches = get_good_matches(dst1, dst2)

    # Get keypoints with good matches
    src_kp, dst_kp = get_final_points(kp1, kp2, good_matches)

    # Find homography
    H, mask = find_homography(src_kp, dst_kp)

    # Transform image with warp perspective
    transformed_image = get_transformed_image(image2, H, final_width)

    return transformed_image

def get_panorama_image(w, h, homography_images):
    """
    Stitches the transformed images with one another
    
    :param w: width of the original image
    :param h: height of the original image
    :param homography_images: list of hmography images returned from above function by combination of 2 images
    """
    l = len(homography_images)

    prev_stitched_image = homography_images[0]

    for image_index in range(1, l):
        homography_images[image_index][0:h, w:] = prev_stitched_image

        prev_stitched_image = homography_images[image_index]

        cv2.imshow("Prev", prev_stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return prev_stitched_image

def main():
    folder_path = input("Enter folder path for panorama:")

    files_path = get_files_list(folder_path, ".png")

    if not files_path:
        return print("Input folder is empty")

    first_image = load_image(files_path[0], True, resize_factor)
    homography_images = [first_image]

    h, w = first_image.shape[:2]

    final_width = w
    final_height = h
    total_images = len(files_path)

    for image_index in range(1, total_images):
        final_width += w

        curr_img_path = files_path[image_index]
        transformed_image = get_homography_image(homography_images[-1], curr_img_path, final_width)
        
        # show_image(transformed_image, "Transformed" + str(image_index))
        # show_image(homography_images[-1], "Previous")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Stitched image {image_index} with {image_index+1}")

        homography_images.append(transformed_image)

    panorama_image = get_panorama_image(w, h, homography_images)

    show_image(panorama_image, "Panorama image")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()