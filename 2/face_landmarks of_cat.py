import dlib
import numpy as np
import cv2

#read image
image_cat = cv2.imread("00000023_020.jpg")

#read cat points and convert into appropriate array
#cat points are inverse of template points (eg: template[x,y] equals cat[y,x]
cat_file = open("00000023_020.jpg.cat", "r")
cat_points = cat_file.read()
cat_points = cat_points.split()
cat_points = np.array(cat_points)
cat_points = cat_points[1:].copy().reshape(-1,2).astype("int")

#mark cat points
for each in cat_points:
    image_cat = cv2.rectangle(image_cat, (each[0] - 2, each[1] - 2), (each[0] + 2, each[1] + 2), (0, 255, 0), -1)

#read template
template_points = np.load("template_points.npy")

#calculate eyes
left_eye = (template_points[36] + template_points[39]) / 2
right_eye = (template_points[42] + template_points[45]) / 2

#apply horizontal ratio
template_points[:,1] = template_points[:,1] * (cat_points[1][0] - cat_points[0][0]) / (right_eye[1] - left_eye[1])

#calculate eye centers
center_eye = (left_eye + right_eye) / 2
center_eye_cat = (cat_points[1] + cat_points[0]) / 2

#apply vertical ratio
template_points[:,0] = template_points[:,0] * (cat_points[2][1] - center_eye_cat[1]) / (template_points[66][0] - center_eye[0])

#convert to integer
template_points = template_points.astype("int")

#move points to appropriate locations
template_points[:,0] = template_points[:,0] - (template_points[66][0] - cat_points[2][1])
template_points[:,1] = template_points[:,1] - (template_points[66][1] - cat_points[2][0])

#mark template points
for each in template_points:
    image_cat = cv2.rectangle(image_cat, (each[1] - 2, each[0] - 2), (each[1] + 2, each[0] + 2), (255, 0, 0), -1)

#show
cv2.imshow("Cat image", image_cat)
cv2.waitKey()