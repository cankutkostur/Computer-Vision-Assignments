import pyautogui
import numpy as np
import cv2
import time

def convolution(image, filter):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # return image
    result = np.zeros(image.shape)
    # zero padding
    image = np.pad(image, 1, mode='constant')

    for i in range(0, image.shape[0]-2):
        for j in range(0, image.shape[1]-2):
            result[i, j] = np.sum(image[i:i+3, j:j+3] * filter)

    #result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return result

def G(x, y, I_x, I_y):
    g = np.zeros((2, 2))

    # create g matrix for given x y
    g[0, 0] = np.sum(np.square(I_x[x - 1:x + 2, y - 1:y + 2])) / 9
    g[0, 1] = np.sum(I_x[x-1:x+2, y-1:y+2] * I_y[x-1:x+2, y-1:y+2]) / 9
    g[1, 0] = g[0, 1]
    g[1, 1] = np.sum(np.square(I_y[x - 1:x + 2, y - 1:y + 2])) / 9

    return g

sobel_y = np.asarray([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]])

sobel_x = np.asarray([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])

# sleep 5 seconds
time.sleep(5)
# take screen shot
screenshot = pyautogui.screenshot()
# resize for faster processing
screenshot = screenshot.resize((1280, 720))
screenshot = np.array(screenshot)
screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

# calculate gradients with sobel filters
screenshot_y = convolution(screenshot.copy(), sobel_y)
screenshot_x = convolution(screenshot.copy(), sobel_x)

# zero paddings
screenshot_x = np.pad(screenshot_x, 1, mode='constant')
screenshot_y = np.pad(screenshot_y, 1, mode='constant')

# keeps corners
corners = []

for x in range(screenshot.shape[0]):
    for y in range(screenshot.shape[1]):
        # min eigenvalue
        min_eigen_value = np.linalg.eigvals(G(x, y, screenshot_x, screenshot_y)).min()

        # thresholding
        if min_eigen_value > 15000:
            # add corner to list
            corners.append((x, y))
            # draw corner
            screenshot = cv2.rectangle(screenshot, (y-2, x-2), (y+2, x+2), (0, 255, 0), -1)

# save image with marked corners
cv2.imwrite("screenshot_corners.png", screenshot)