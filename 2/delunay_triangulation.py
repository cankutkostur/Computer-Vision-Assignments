import dlib
import numpy as np
import cv2

def get_cat_landmarks():
    # read cat points and convert into appropriate array
    # cat points are inverse of template points (eg: template[x,y] equals cat[y,x]
    cat_file = open("00000023_020.jpg.cat", "r")
    cat_points = cat_file.read()
    cat_points = cat_points.split()
    cat_points = np.array(cat_points)
    cat_points = cat_points[1:].copy().reshape(-1, 2).astype("int")

    # read template
    template_points = np.load("template_points.npy")

    # calculate eyes
    left_eye = (template_points[36] + template_points[39]) / 2
    right_eye = (template_points[42] + template_points[45]) / 2

    # apply horizontal ratio
    template_points[:, 1] = template_points[:, 1] * (cat_points[1][0] - cat_points[0][0]) / (right_eye[1] - left_eye[1])

    # calculate eye centers
    center_eye = (left_eye + right_eye) / 2
    center_eye_cat = (cat_points[1] + cat_points[0]) / 2

    # apply vertical ratio
    template_points[:, 0] = template_points[:, 0] * (cat_points[2][1] - center_eye_cat[1]) / (
                template_points[66][0] - center_eye[0])

    # convert to integer
    template_points = template_points.astype("int")

    # move points to appropriate locations
    template_points[:, 0] = template_points[:, 0] - (template_points[66][0] - cat_points[2][1])
    template_points[:, 1] = template_points[:, 1] - (template_points[66][1] - cat_points[2][0])

    return template_points

def insert_edges(subdiv, image):
    subdiv.insert((0, 0))
    subdiv.insert((0, int(image.shape[1] / 2)))
    subdiv.insert((0, image.shape[1] - 1))
    subdiv.insert((image.shape[0] - 1, 0))
    subdiv.insert((image.shape[0] - 1, int(image.shape[1] / 2)))
    subdiv.insert((image.shape[0] - 1, image.shape[1] - 1))
    subdiv.insert((int(image.shape[0] / 2), 0))
    subdiv.insert((int(image.shape[0] / 2), image.shape[1] - 1))

def draw_triangles(image, triangles):
    for i in range(len(triangles)):
        sel_triangle = triangles[i].astype(np.int)
        cv2.line(image, (sel_triangle[1], sel_triangle[0]), (sel_triangle[3], sel_triangle[2]), (0, 255, 0), 1)
        cv2.line(image, (sel_triangle[1], sel_triangle[0]), (sel_triangle[5], sel_triangle[4]), (0, 255, 0), 1)
        cv2.line(image, (sel_triangle[5], sel_triangle[4]), (sel_triangle[3], sel_triangle[2]), (0, 255, 0), 1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#read images
image_cat = cv2.imread("00000023_020.jpg")

image_1 = cv2.imread("input1.jpg")
gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

image_2 = cv2.imread("input2.jpg")
gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

#predict rectangles
rectangles_1 = detector(gray_1)
rectangles_2 = detector(gray_2)

#predict points
points_1 = predictor(gray_1, rectangles_1[0])
points_2 = predictor(gray_2, rectangles_2[0])
points_cat = get_cat_landmarks()

subdiv_1 = cv2.Subdiv2D((0, 0, image_1.shape[0], image_1.shape[1]))
subdiv_2 = cv2.Subdiv2D((0, 0, image_2.shape[0], image_1.shape[1]))
subdiv_cat = cv2.Subdiv2D((0, 0, image_cat.shape[0], image_cat.shape[1]))

for i in range(68):
    subdiv_1.insert((points_1.part(i).y, points_1.part(i).x))
    subdiv_2.insert((points_2.part(i).y, points_2.part(i).x))
    subdiv_cat.insert((points_cat[i, 0], points_cat[i, 1]))

#insert edges
insert_edges(subdiv_1, image_1)
insert_edges(subdiv_2, image_2)
insert_edges(subdiv_cat, image_cat)

#get triangles
triangles_1 = subdiv_1.getTriangleList()
triangles_2 = subdiv_2.getTriangleList()
triangles_cat = subdiv_cat.getTriangleList()

#draw triangles
draw_triangles(image_1, triangles_1)
draw_triangles(image_2, triangles_2)
draw_triangles(image_cat, triangles_cat)

final_image = np.concatenate((image_2, image_1, image_cat), axis=1)

cv2.imshow("Image", final_image)
cv2.waitKey()