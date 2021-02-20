import numpy as np
import cv2
import glob
from moviepy.editor import *

video_names = ["walking", "mbike-trick", "bike-packing"]

for video_num in range(len(video_names)):
    all_images = glob.glob(("DAVIS\\JPEGImages\\" + video_names[video_num] + "\\*.jpg"))  # reads all frames
    all_seg = glob.glob("DAVIS\\Annotations\\" + video_names[video_num] + "\\*.png")

    images_list = []  # keeps video frames

    for num in range(len(all_images)):
        image = cv2.imread(all_images[num])
        seg = cv2.imread(all_seg[num], cv2.IMREAD_GRAYSCALE)

        seg = np.array(seg == 38)  # segmentation binary map for 38
        image_of_the_guy = cv2.bitwise_and(image, image, mask=seg.astype(np.uint8))
        #for i in range(3):  # applying seg filter for all channels
        #    image_of_the_guy[:, :, i] *= seg

        seg = np.array(seg == False)  # segmentation binary map for not 38
        image_without_the_guy = cv2.bitwise_and(image, image, mask=seg.astype(np.uint8))
        #for i in range(3):  # applying seg filter for all channels
        #    image_without_the_guy[:, :, i] *= seg

        image_of_the_guy[:, :, 1:3] = np.array(image_of_the_guy[:, :, 1:3]) / 4  # degrading red and green channels by 75%

        image = image_of_the_guy + image_without_the_guy  # merging guy and rest of image
        image_rgb = image[..., ::-1]  # converting bgr to rgb

        images_list.append(image_rgb)

    clip = ImageSequenceClip(images_list, fps=25)
    clip.write_videofile("part1_video" + str(video_num + 1) + ".mp4", codec="mpeg4")
