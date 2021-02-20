import numpy as np
import cv2
from moviepy.editor import *

image = np.zeros((500, 394, 3), dtype='uint8')

#determşne triangles
source_points = np.array([[160, 160], [220, 200], [350, 150]])
target_points = np.array([[180, 250], [220, 320], [320, 360]])

M = cv2.getAffineTransform(source_points.astype('float32'), target_points.astype('float32'))
#to keep source triangle
M[0][0] -= 1
M[1][1] -= 1
M /= 20

frames = []

for i in range(21):
    frame = np.zeros((500, 394, 3), dtype='uint8')

    cv2.polylines(image, [source_points], isClosed=True, color=(i/20 * 255, 0, (20-i)/20 * 255), thickness=1)
    cv2.fillPoly(image, [source_points], color=(i/20 * 255, 0, (20-i)/20 * 255))

    M_frame = M*i
    #to add source triangle that we subrtacted before
    M_frame[0][0] += 1
    M_frame[1][1] += 1
    frame = cv2.warpAffine(image, M_frame, (394, 500))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

clip = ImageSequenceClip(frames, fps=20)
clip.write_videofile("part4.mp4", codec="mpeg4")
