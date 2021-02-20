import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#read images
image_1 = cv2.imread("input1.jpg")
gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

image_2 = cv2.imread("input2.jpg")
gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

image_cat = cv2.imread("00000023_020.jpg")

#predict rectangles
rectangles_1 = detector(gray_1)
rectangles_2 = detector(gray_2)

rectangle_color = (255, 0, 0)

#draw face rectangles
image_1 = cv2.rectangle(image_1, (rectangles_1[0].tl_corner().x, rectangles_1[0].tl_corner().y), (rectangles_1[0].br_corner().x, rectangles_1[0].br_corner().y), rectangle_color, 2)
image_2 = cv2.rectangle(image_2, (rectangles_2[0].tl_corner().x, rectangles_2[0].tl_corner().y), (rectangles_2[0].br_corner().x, rectangles_2[0].br_corner().y), rectangle_color, 2)

#predict points
points_1 = predictor(gray_1, rectangles_1[0])
points_2 = predictor(gray_2, rectangles_2[0])

mark_color = (0, 255, 0)

#mark points
for i in range(68):
    image_1 = cv2.rectangle(image_1, (points_1.part(i).x-2, points_1.part(i).y-2), (points_1.part(i).x+2, points_1.part(i).y+2), mark_color, -1)
    image_2 = cv2.rectangle(image_2, (points_2.part(i).x-2, points_2.part(i).y-2), (points_2.part(i).x+2, points_2.part(i).y+2), mark_color, -1)

#cat points
cat_file = open("00000023_020.jpg.cat", "r")
cat_points = cat_file.read()
cat_points = cat_points.split()

#mark cat points
for i in range(9):
    x = int(cat_points[2*i + 1])
    y = int(cat_points[2*i + 2])
    image_cat = cv2.rectangle(image_cat, (x-2, y-2), (x+2, y+2), mark_color, -1)

#merge
final_image = np.concatenate((image_2, image_1, image_cat), axis=1)

#show
cv2.imshow("Image Final", final_image)
cv2.waitKey()