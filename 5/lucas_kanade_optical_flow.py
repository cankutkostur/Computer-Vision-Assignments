import cv2
import numpy as np
import moviepy.video.io.VideoFileClip as mpy
from moviepy.editor import ImageSequenceClip


def gradients(img1, img2):
    I_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    I_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
    I_t = img2 - img1

    return I_x, I_y, I_t


def optical_flow(I_x, I_y, I_t):
    ATA = [np.sum(I_x*I_x), np.sum(I_x*I_y)], [np.sum(I_x*I_y), np.sum(I_y*I_y)]
    ATA = np.asarray(ATA)

    ATb = [np.sum(I_x*I_t), np.sum(I_y*I_t)]
    ATb = np.asarray(ATb)

    V = np.matmul(-np.linalg.inv(ATA), ATb)

    return V


def lucas_kanade(frame1, frame2, point, k=9):
    I_x, I_y, I_t = gradients(frame1, frame2)

    I_x = I_x[point[1] - int(k / 2):point[1] + int(k / 2) + 1, point[0] - int(k / 2):point[0] + int(k / 2) + 1]
    I_y = I_y[point[1] - int(k / 2):point[1] + int(k / 2) + 1, point[0] - int(k / 2):point[0] + int(k / 2) + 1]
    I_t = I_t[point[1] - int(k / 2):point[1] + int(k / 2) + 1, point[0] - int(k / 2):point[0] + int(k / 2) + 1]

    return optical_flow(I_x, I_y, I_t)


walker = mpy.VideoFileClip("walker.avi")
walker_hand = mpy.VideoFileClip("walker_hand.avi")

frame_count = walker_hand.reader.nframes
video_fps = walker_hand.fps

frames = []

for i in range(frame_count):
    if i % 2 == 0:
        current_frame = walker.get_frame(i*1.0/video_fps)
        current_frame_grey = cv2.cvtColor(np.float32(current_frame), cv2.COLOR_BGR2GRAY)
        mask = walker_hand.get_frame(i*1.0/video_fps)

        next_frame = walker.get_frame((i+2)*1.0/video_fps)
        next_frame_grey = cv2.cvtColor(np.float32(next_frame), cv2.COLOR_BGR2GRAY)

        current_frame_grey = cv2.GaussianBlur(current_frame_grey, (3, 3), 0)
        next_frame_grey = cv2.GaussianBlur(next_frame_grey, (3, 3), 0)

        mask = (mask > 127)
        current_frame_hand = mask * current_frame
        hand_grey = cv2.cvtColor(np.float32(current_frame_hand), cv2.COLOR_BGR2GRAY)

        corners_hand = np.array(cv2.goodFeaturesToTrack(hand_grey, 25, 0.01, 5)).astype('int')

        arrows = []
        for corner in corners_hand:
            # multiply with 60 because 30 fps and we skip every odd frame
            vector = (lucas_kanade(current_frame_grey, next_frame_grey, corner[0]) * 60)
            start = (corner[0][0], corner[0][1])
            end = (int(corner[0][0] + vector[0]), int(corner[0][1] + vector[1]))

            arrows.append((start, end))

        # averaging arrows
        arrow_start = [0, 0]
        arrow_end = [0, 0]
        for a in arrows:
            arrow_start[0] += a[0][0]
            arrow_start[1] += a[0][1]
            arrow_end[0] += a[1][0]
            arrow_end[1] += a[1][1]

        arrow_start[0] /= len(arrows)
        arrow_start[1] /= len(arrows)
        arrow_end[0] /= len(arrows)
        arrow_end[1] /= len(arrows)

        current_frame = cv2.arrowedLine(current_frame, (int(arrow_start[0]), int(arrow_start[1])), (int(arrow_end[0]), int(arrow_end[1])), (255, 0, 0), 2)

        frames.append(current_frame)

clip = ImageSequenceClip(frames, fps=15)
clip.write_videofile("part1.mp4", codec='mpeg4')




