# script to look at all data with opencv

import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt

from kachow import CircleOfExpansionEstimator

plt.ion()

data_folder = "data/first_lap/rgb"

# sort by the number in the file name
files = sorted(os.listdir(data_folder), key=lambda x: int(x.split("_")[1].split(".")[0]))

circle_center_list = []
for i in range(0, len(files)):
    file = files[i]
    print(file)
    img = cv2.imread(f"{data_folder}/{file}")
    
    start = time.time()

    lines_img = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = img[:, : , 1]
    edges = cv2.Canny(gray, 30, 50, apertureSize=3, L2gradient=False)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=25)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # find slope of line
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 1:# and abs(slope) > 0.1:
                #     # check length of line

            # draw lines through the points
                    intercept = y1 - slope * x1
                    cv2.line(lines_img, (0, int(intercept)), (img.shape[1], int(slope * img.shape[1] + intercept)), (0, 255, 0), 2)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    
    end = time.time()
    print(f"Time: {end - start} seconds")

    cv2.imshow("gray", gray)
    cv2.imshow("edges", edges)
    cv2.imshow("lines", lines_img)
    cv2.imshow("img", img)
    # if circle_center is not None:
    #     circle_center_list.append(circle_center[0])
    # plt.clf()
    # plt.plot(circle_center_list)

    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
        break