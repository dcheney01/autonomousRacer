from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm
from torchsummary import summary
import torch

# Load the exported TensorRT model
weights = "/home/curtis/classes/robotic_vision/autonomousRacer/segmentation/best.pt"

segmenter = YOLO(weights, task='segmentation')

#save state dict
torch.save(segmenter.model.state_dict(), 'best.pth')

#print model to text file
with open('ultralytics.txt', 'w') as f:
    print(segmenter.model, file=f)
# summary(segmenter.model.model.cuda(), (3, 640, 480))
image_path = "./fourth_lap/rgb/00011.jpg"
img = cv2.imread(image_path)
# img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
H, W, _ = img.shape
print(img.shape)

# # Run inference
results = segmenter(img)
# print(results[0].shape)

# for result in results:
#     for j, mask in enumerate(result.masks.data):
#         mask = mask.cpu().numpy() * 255

#         mask = cv2.resize(mask, (W, H))
#         opacity = 0.3
#         cv2.addWeighted(mask, opacity, img, 1 - opacity, 0, img)

#         cv2.imwrite(f'./inference/{image_path}.png', mask)

#     # #display mask on top of img with opacity
#     # mask = cv2.imread('./output.png', cv2.IMREAD_GRAYSCALE)
#     # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#     # opacity = 0.3
#     # cv2.addWeighted(mask, opacity, img, 1 - opacity, 0, img)
#     # cv2.imshow('img', img)
#     # key = cv2.waitKey(0)
#     # if key == 'q':
#     #     cv2.destroyAllWindows()

# import onnxruntime as ort
# import numpy as np
# import cv2

# onnx_model = ort.InferenceSession("./yolov8n.onnx")

# def predict(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)

#     image = np.expand_dims(image, axis=0).astype('float32') / 255.
#     image = np.transpose(image, [0, 3, 1, 2])
#     outputs = onnx_model.run(None, {'images': image})
#     return outputs

# img = cv2.imread("./fourth_lap/rgb/00011.jpg")
# results = predict(img)
# print(results)
# print(results[0].shape)

# for result in results:
#     for j, mask in enumerate(result.masks.data):
#         print(mask.shape)
#         print(img.shape)
#         mask = mask.cpu().numpy() * 255

#         mask = cv2.resize(mask, (W, H))
#         opacity = 0.3
#         cv2.addWeighted(mask, opacity, img, 1 - opacity, 0, img)

#         cv2.imwrite(f'./inference/test.png', mask)
