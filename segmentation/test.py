from ultralytics import YOLO
import cv2
import os

model = YOLO('./runs/segment/train4/weights/best.pt')  # Load a model

image_paths = os.listdir("../data/fourth_lap/rgb/")
image_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for img in image_paths:
    image = cv2.imread("../data/fourth_lap/rgb/" + img)
    image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("data/fourth_lap/rgb/" + img)
    result = model.predict(source=image, show=True, verbose=False)