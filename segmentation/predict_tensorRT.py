from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm

# Load the exported TensorRT model
tensorrt_model = YOLO(
    '/home/curtis/classes/robotic_vision/racing/runs/segment/train2/weights/best.engine'
)

image_dir = '/home/curtis/classes/robotic_vision/racing/fourth_lap/rgb/'
images = os.listdir(
    '/home/curtis/classes/robotic_vision/racing/fourth_lap/rgb/')

for image_path in tqdm(images):
    print(image_path)
    img = cv2.imread(image_dir + image_path)
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    H, W, _ = img.shape
    print(img.shape)

    # Run inference
    results = tensorrt_model.predict(img, imgsz=160)

    for result in results:
        for j, mask in enumerate(result.masks.data):
            print(mask.shape)
            print(img.shape)
            mask = mask.cpu().numpy() * 255

            mask = cv2.resize(mask, (W, H))
            opacity = 0.3
            cv2.addWeighted(mask, opacity, img, 1 - opacity, 0, img)

            cv2.imwrite(f'./inference/{image_path}.png', mask)

    # #display mask on top of img with opacity
    # mask = cv2.imread('./output.png', cv2.IMREAD_GRAYSCALE)
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # opacity = 0.3
    # cv2.addWeighted(mask, opacity, img, 1 - opacity, 0, img)
    # cv2.imshow('img', img)
    # key = cv2.waitKey(0)
    # if key == 'q':
    #     cv2.destroyAllWindows()
