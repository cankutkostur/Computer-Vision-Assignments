import numpy as np
import dlib
from moviepy.editor import *
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

def get_triangles(triangles, points_1, points_2):
    to_ret = []
    for each in triangles:
        triangle = []
        for i in range(3):
            for j in range(68):
                #corresponing ids locations added
                if points_1.part(j).y == each[i*2] and points_1.part(j).x == each[i*2+1]:
                    triangle.append(points_2.part(j).y)
                    triangle.append(points_2.part(j).x)
                    break
                if j == 67: #if border triangle
                    triangle.append(int(each[i * 2]))
                    triangle.append(int(each[i * 2 + 1]))

        triangle = np.asarray(triangle)
        to_ret.append(triangle)

    to_ret = np.asarray(to_ret)
    return to_ret

#same for cat
def get_triangles_cat(triangles, points_1, points_2):
    to_ret = []
    for each in triangles:
        triangle = []
        for i in range(3):
            for j in range(68):
                if points_1.part(j).y == each[i*2] and points_1.part(j).x == each[i*2+1]:
                    triangle.append(points_2[j][0])
                    triangle.append(points_2[j][1])
                    break
                if j == 67:
                    triangle.append(int(each[i * 2]))
                    triangle.append(int(each[i * 2 + 1]))

        triangle = np.asarray(triangle)
        to_ret.append(triangle)

    to_ret = np.asarray(to_ret)
    return to_ret

def morph(image_1, image_2, triangles_1, triangles_2):
    M_1 = []
    M_2 = []

    for i in range(len(triangles_1)):
        #fliplrs used for inverse x y coordinates
        triangle_1 = triangles_1[i]
        triangle_1 = triangle_1.reshape((3, 2))
        triangle_1 = np.fliplr(triangle_1)
        triangle_2 = triangles_2[i]
        triangle_2 = triangle_2.reshape((3, 2))
        triangle_2 = np.fliplr(triangle_2)

        m_1 = cv2.getAffineTransform(triangle_1.astype('float32'), triangle_2.astype('float32'))
        #keep source
        m_1[0][0] -= 1
        m_1[1][1] -= 1
        m_1 /= 20
        M_1.append(m_1)

        m_2 = cv2.getAffineTransform(triangle_2.astype('float32'), triangle_1.astype('float32'))
        #keep source
        m_2[0][0] -= 1
        m_2[1][1] -= 1
        m_2 /= 20
        M_2.append(m_2)

    frames = []

    for i in range(21):
        alpha = i / 20

        frame = np.zeros(image_1.shape, dtype='uint8')

        for j in range(len(triangles_1)):
            #calculate transform for this step
            m_1 = M_1[j] * i
            #add source subtracted before
            m_1[0][0] += 1
            m_1[1][1] += 1

            # calculate transform for this step
            m_2 = M_2[j] * (20 - i)
            # add source subtracted before
            m_2[0][0] += 1
            m_2[1][1] += 1

            triangle_1 = triangles_1[j]
            triangle_1 = triangle_1.reshape((3, 2))
            triangle_1 = np.fliplr(triangle_1)
            triangle_2 = triangles_2[j]
            triangle_2 = triangle_2.reshape((3, 2))
            triangle_2 = np.fliplr(triangle_2)
            triangle_t = ((1 - alpha) * triangle_1 + alpha * triangle_2)
            triangle_t = triangle_t.astype('int32')

            #mask for this triangle
            mask = np.zeros(image_1.shape, dtype='uint8')
            cv2.fillConvexPoly(mask, triangle_t, (1, 1, 1), 16, 0)
            #inverse mask for rest of image to eliminate border issues
            mask_inverse = mask.copy()
            mask_inverse[mask == 0] = 1
            mask_inverse[mask == 1] = 0

            #apply transforms and masks
            frame_1 = cv2.warpAffine(image_1, m_1, (image_1.shape[1], image_1.shape[0]), borderMode=cv2.BORDER_TRANSPARENT)
            frame_1 *= mask
            frame_2 = cv2.warpAffine(image_2, m_2, (image_1.shape[1], image_1.shape[0]), borderMode=cv2.BORDER_TRANSPARENT)
            frame_2 *= mask

            frame *= mask_inverse
            frame += ((1 - alpha) * frame_1 + alpha * frame_2).astype('uint8')

        #add each frame to frames list as rgb
        frame = cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_BGR2RGB)
        frames.append(frame)


    return frames


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

#predict points
points_1 = predictor(gray_1, rectangles_1[0])
points_2 = predictor(gray_2, rectangles_2[0])
points_cat = get_cat_landmarks()

subdiv_1 = cv2.Subdiv2D((0, 0, image_1.shape[0], image_1.shape[1]))

for i in range(68):
    subdiv_1.insert((points_1.part(i).y, points_1.part(i).x))

insert_edges(subdiv_1, image_1)

#get triangles
triangles_1 = subdiv_1.getTriangleList()
triangles_2 = get_triangles(triangles_1, points_1, points_2)
triangles_cat = get_triangles_cat(triangles_1, points_1, points_cat)

#store in lists for iterate
images = [image_1, image_2, image_cat]
triangles = [triangles_1, triangles_2, triangles_cat]

#create videos for all permutations
for i in range(2):
    for j in range(i+1, 3):
        frames = morph(images[i], images[j], triangles[i], triangles[j])

        clip = ImageSequenceClip(frames, fps=20)
        clip.write_videofile("part5_" + str(i+1) + "_to_" + str(j+1) + ".mp4", codec="mpeg4")

        frames = morph(images[j], images[i], triangles[j], triangles[i])

        clip = ImageSequenceClip(frames, fps=20)
        clip.write_videofile("part5_" + str(j+1) + "_to_" + str(i+1) + ".mp4", codec="mpeg4")