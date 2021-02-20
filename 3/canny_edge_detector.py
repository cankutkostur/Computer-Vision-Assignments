import time
import pyautogui
import cv2
import numpy as np


# sleep 5 seconds
time.sleep(5)

# take screen shot
screenshot = pyautogui.screenshot()

# resize for faster processing
screenshot = screenshot.resize((1280, 720))

# convert to numpy array
screenshot = np.array(screenshot)
screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

# convert to greyscale
screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

# find edges using cv2's canny
edges = cv2.Canny(screenshot_gray, 150, 200)

contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(screenshot, contours, -1, (0, 255, 0), 3)
cv2.imshow('contours', screenshot)
cv2.waitKey()

# save edges
cv2.imwrite('screenshot_edges_canny.png', edges)