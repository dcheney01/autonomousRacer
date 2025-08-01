#resize all images to 25% of their original size
import os
import cv2
from tqdm import tqdm

input_dir = './tmp/labels/lap3/'
output_dir = './tmp/labels/lap3_resized/'

os.makedirs(output_dir, exist_ok=True)

for j in tqdm(os.listdir(input_dir)):
    image_path = os.path.join(input_dir, j)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, j), image)
