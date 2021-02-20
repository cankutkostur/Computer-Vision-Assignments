import cv2
import numpy as np


def main():
    # Threshold
    T = 150

    # load image
    image = cv2.imread("marker.png")
    # resize image to speed up process
    width = image.shape[1] / 2
    height = image.shape[0] / 2
    dim = (int(width), int(height))
    image = cv2.resize(image, dim)

    image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect edge points using Canny
    edges = cv2.Canny(image_greyscale, 150, 200)
    # create lists of edge points
    edges_coordinates = np.where(edges == 255)
    edges_coordinates = list(zip(edges_coordinates[0], edges_coordinates[1]))
    # arrange angles
    theta = np.arange(0, 360) * np.pi / 180

    # try with various radius
    for r in range(10, 11):
        # define accumulator array and fill with zeros
        acc = np.zeros_like(image_greyscale)
        # iterate over edge coordinates
        for x, y in edges_coordinates:
            # iterate for every angle
            for angle in theta:
                a = x - int(round(r * np.cos(angle)))
                b = y - int(round(r * np.sin(angle)))
                # increments point count for current parameter values
                if 0 <= a < acc.shape[0] and 0 <= b < acc.shape[1]:
                    acc[a][b] += 1

        # check if there is any circle contains more point than T
        if acc.max() > T:
            # create centers list for circles has more point than T
            centers = np.where(acc > T)
            centers = list(zip(centers[0], centers[1]))

            # iterate over centers and draw circles
            for a, b in centers:
                cv2.circle(image, (b, a), r, (0, 0, 255), 2)
                # for angle in theta:
                #     x = a + int(round(r * np.cos(angle)))
                #     y = b + int(round(r * np.sin(angle)))
                #     if edges[x][y] == 255:
                #         image[x][y] = (0, 0, 255)

    # show image with circles
    cv2.imshow("circles", image)
    cv2.waitKey()

    cv2.imwrite("circles.png", image)


if __name__ == "__main__":
    main()
