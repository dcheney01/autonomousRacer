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

coeEstimator = CircleOfExpansionEstimator(display=True)

circle_center_list = []
for i in range(0, len(files)):
    file = files[i]
    print(file)
    img = cv2.imread(f"{data_folder}/{file}")
    
    start = time.time()

    # circle_center = coeEstimator.get_circle_of_expansion(img) # add force new features option, rename the class to something else
    circle_center = coeEstimator.get_features_centerpoint(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    end = time.time()
    print(f"Time: {end - start} seconds")

    # if circle_center is not None:
    #     circle_center_list.append(circle_center[0])
    # plt.clf()
    # plt.plot(circle_center_list)

    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
        break