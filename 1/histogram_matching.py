import numpy as np
import cv2
import glob
from moviepy.editor import *
import matplotlib.pyplot as plt

def histogram(I):
    R, C, B = I.shape

    hist = np.zeros([256, 1, B])

    for g in range(256):
        hist[g, 0, ...] = np.sum(np.sum(I == g, 0), 0)

    return hist

def cdf_from_pdf(p):
    P = p
    for i in range(1, 256):
        P[i] += P[i - 1]

    return P;

video_names = ["walking", "mbike-trick", "bike-packing"]
target_image_names = ["target1.jpg", "target1.jpg", "target1.jpg"]

for video_num in range(len(video_names)):
    all_images = glob.glob(("DAVIS\\JPEGImages\\" + video_names[video_num] + "\\*.jpg"))  # reads all frames
    all_seg = glob.glob("DAVIS\\Annotations\\" + video_names[video_num] + "\\*.png")

    images_list = []  # keeps video frames

    target_image = cv2.imread(target_image_names[video_num])
    target_r, target_c, target_b = target_image.shape
    target_pdf = histogram(target_image) / (target_r * target_c)
    target_cdf = cdf_from_pdf(target_pdf)

    video_pdf = np.zeros([256, 1, 3])
    for i in range(len(all_images)):  # histogram calculating for all frames
        image = cv2.imread(all_images[i])
        image_histogram = histogram(image)
        video_pdf += image_histogram * 3 / np.sum(image_histogram)

    video_pdf /= len(all_images)

    video_cdf = cdf_from_pdf(video_pdf)

    LUT = np.zeros((256, 1, 3))
    for i in range(3):  # for all channels
        g_j = 0
        for g_i in range(256):  # creating lut table for histogram matching
            while target_cdf[g_j, 0, i] < video_cdf[g_i, 0, i] and g_j < 255:
                g_j += 1

            LUT[g_i, 0, i] = g_j

    for i in range(len(all_images)):
        image = cv2.imread(all_images[i])
        R, C, B = image.shape
        for r in range(R):  # point wise lut
            for c in range(C):
                for b in range(B):
                    image[r, c, b] = LUT[image[r, c, b], 0, b]
        image_rgb = image[..., ::-1]  # converting bgr to rgb
        images_list.append(image_rgb)  # append frame for video

    clip = ImageSequenceClip(images_list, fps=25)
    clip.write_videofile("part2_video" + str(video_num + 1) + ".mp4", codec="mpeg4")