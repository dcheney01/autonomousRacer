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

for file in files:
    print(file)
    img = cv2.imread(f"{data_folder}/{file}")
    
    start = time.time()

    circle_center = coeEstimator.get_circle_of_expansion(img)
    
    end = time.time()
    print(f"Time: {end - start} seconds")

    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
        break