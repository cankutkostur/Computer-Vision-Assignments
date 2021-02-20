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


def histogram_seperated(I, seg):
    R, C, B = I.shape

    hist = np.zeros([256, 1, B])

    for g in range(1, 256):  # starts from 1 for not counting non-object pixels
        hist[g, 0, ...] = np.sum(np.sum(I == g , 0), 0)

    for b in range(B):  # calculates 0 pixel of the object by simply substracting counted pixel count from total pixel count of the object
        hist[0, 0, b] = np.sum(seg) - np.sum(hist[:, 0, b])

    return hist


def cdf_from_pdf(p):
    P = np.copy(p)
    for i in range(1, 256):
        P[i] += P[i - 1]

    return P;


def lut_table(video_cdf, target_cdf):
    LUT = np.zeros((256, 1, 3))
    for i in range(3):  # for all channels
        g_j = 0
        for g_i in range(256):  # creating lut table for histogram matching
            while target_cdf[g_j, 0, i] < video_cdf[g_i, 0, i] and g_j < 255:
                g_j += 1

            LUT[g_i, 0, i] = g_j

    return LUT

def lut_transform(image, LUT, seg):
    R, C, B = image.shape
    for r in range(R):  # point wise lut
        for c in range(C):
            if seg[r, c]:  # to apply only if it is that objects pixel
                for b in range(B):
                    image[r, c, b] = LUT[image[r, c, b], 0, b]

    return image


video_names = ["bmx-trees", "mbike-trick", "bike-packing"]
target_image_names = ["target1.jpg", "target2.jpg", "target3.jpg"]

for video_num in range(len(video_names)):
    all_images = glob.glob(("DAVIS\\JPEGImages\\" + video_names[video_num] + "\\*.jpg"))  # reads all frames
    all_seg = glob.glob("DAVIS\\Annotations\\" + video_names[video_num] + "\\*.png")

    images_list = []  # keeps video frames

    target1 = cv2.imread(target_image_names[0])
    target1_histogram = histogram(target1)
    target1_pdf = target1_histogram * 3 / np.sum(target1_histogram)
    target1_cdf = cdf_from_pdf(target1_pdf)

    target2 = cv2.imread(target_image_names[1])
    target2_histogram = histogram(target2)
    target2_pdf = target2_histogram * 3 / np.sum(target2_histogram)
    target2_cdf = cdf_from_pdf(target2_pdf)

    target3 = cv2.imread(target_image_names[2])
    target3_histogram = histogram(target3)
    target3_pdf = target3_histogram * 3 / np.sum(target3_histogram)
    target3_cdf = cdf_from_pdf(target3_pdf)

    video_pdf_obj1 = np.zeros([256, 1, 3])
    video_pdf_obj2 = np.zeros([256, 1, 3])
    video_pdf_background = np.zeros([256, 1, 3])

    for i in range(len(all_images)):  # histogram calculating for all frames
        image = cv2.imread(all_images[i])
        seg = cv2.imread(all_seg[i], cv2.IMREAD_GRAYSCALE)

        seg_obj1 = (seg == 38)
        obj1 = cv2.bitwise_and(image, image, mask=seg_obj1.astype(np.uint8))

        seg_obj2 = (seg == 75)
        obj2 = cv2.bitwise_and(image, image, mask=seg_obj2.astype(np.uint8))

        seg_background = (seg == 0)
        background = cv2.bitwise_and(image, image, mask=seg_background.astype(np.uint8))

        obj1_histogram = histogram_seperated(obj1, seg_obj1)
        video_pdf_obj1 += obj1_histogram / np.sum(seg_obj1)

        obj2_histogram = histogram_seperated(obj2, seg_obj2)
        video_pdf_obj2 += obj2_histogram / np.sum(seg_obj2)

        background_histogram = histogram_seperated(background, seg_background)
        video_pdf_background += background_histogram / np.sum(seg_background)

    video_pdf_obj1 /= len(all_images)
    video_pdf_obj2 /= len(all_images)
    video_pdf_background /= len(all_images)

    video_cdf_obj1 = cdf_from_pdf(video_pdf_obj1)
    video_cdf_obj2 = cdf_from_pdf(video_pdf_obj2)
    video_cdf_background = cdf_from_pdf(video_pdf_background)

    LUT_obj1 = lut_table(video_cdf_obj1, target1_cdf)
    LUT_obj2 = lut_table(video_cdf_obj2, target2_cdf)
    LUT_background = lut_table(video_cdf_background, target3_cdf)

    for i in range(len(all_images)):
        image = cv2.imread(all_images[i])
        seg = cv2.imread(all_seg[i], cv2.IMREAD_GRAYSCALE)

        seg_obj1 = (seg == 38)
        obj1 = cv2.bitwise_and(image, image, mask=seg_obj1.astype(np.uint8))

        seg_obj2 = (seg == 75)
        obj2 = cv2.bitwise_and(image, image, mask=seg_obj2.astype(np.uint8))

        seg_background = (seg == 0)
        background = cv2.bitwise_and(image, image, mask=seg_background.astype(np.uint8))

        obj1 = lut_transform(obj1, LUT_obj1, seg_obj1)
        obj2 = lut_transform(obj2, LUT_obj2, seg_obj2)
        background = lut_transform(background, LUT_background, seg_background)

        image = obj1 + obj2 + background

        image_rgb = image[..., ::-1]  # converting bgr to rgb
        images_list.append(image_rgb)  # append frame for video

    clip = ImageSequenceClip(images_list, fps=25)
    clip.write_videofile("part3_video" + str(video_num + 1) + ".mp4", codec="mpeg4")
