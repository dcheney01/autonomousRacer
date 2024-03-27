# run this program on the Mac to display image streams from multiple RPis
import cv2
import imagezmq
from ultralytics import YOLO

image_hub = imagezmq.ImageHub()
model = YOLO('./segmentation/runs/segment/train2/weights/best.pt')  # Load a model
print("Server started. Waiting for images...")

j = 0
while True:  # show streamed images until Ctrl-C
    rpi_name, image = image_hub.recv_image()
    # cv2.imshow(rpi_name, image) # 1 window for each RPi
    result = model.predict(source=image, show=True, verbose=False)

    # cv2.waitKey(1)

    image_hub.send_reply(b'OK')
    j += 1