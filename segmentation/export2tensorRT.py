from ultralytics import YOLO

import cv2
import os
from tqdm import tqdm

model_path = '/home/daniel/software/autonomousRacer/segmentation/runs/segment/train2/weights/best.pt'

# image_path = '/home/curtis/classes/robotic_vision/racing/first_lap/rgb/00261.jpg'

# img = cv2.imread(image_path)
# img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
# H, W, _ = img.shape

model = YOLO(model_path)

# #export to tensorrt for speedup on nvidia gpu
success = model.export(format='engine', device='0,1', half=True, imgsz=640)
# files = os.listdir('/home/curtis/classes/robotic_vision/racing/second_lap/rgb/')

# for file in tqdm(files):
#     img = cv2.imread('/home/curtis/classes/robotic_vision/racing/second_lap/rgb/' + file)
#     img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
#     H, W, _ = img.shape


#     results = model(img)

#     for result in results:
#         for j, mask in enumerate(result.masks.data):

#             mask = mask.cpu().numpy() * 255

#             mask = cv2.resize(mask, (W, H))

#             # cv2.imwrite('./output.png', mask)

#     #display mask on top of img with opacity
#     # mask = cv2.imread('./output.png', cv2.IMREAD_GRAYSCALE)
#     # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#     opacity = 0.3
#     cv2.addWeighted(mask, opacity, img, 1 - opacity, 0, img)
#     img = cv2.resize(img, (0, 0), fx=4, fy=4)
#     cv2.imwrite(f'./inference/{file}.png', img)
#     # cv2.imshow('img', img)
#     # key = cv2.waitKey(0)
#     # if key == 'q':
#     #     cv2.destroyAllWindows()
