# run this program on each RPi to send a labelled image stream
# import socket
# import time
# from imutils.video import VideoStream
# import imagezmq
# from RealSense import *

# sender = imagezmq.ImageSender(connect_to='tcp://10.32.114.243:5555')
# rs = RealSense(RS_VGA)		# RS_VGA, RS_720P, or RS_1080P
# (time_, rgb, depth, accel, gyro) = rs.getData()

# host_name = socket.gethostname() # send RPi hostname with each image

# time.sleep(2.0)  # allow camera sensor to warm up

# while True:  # send images as stream until Ctrl-C
#     (time_, img, depth, accel, gyro) = rs.getData()
#     reply_from_server = sender.send_image(host_name, img)
#     print(reply_from_server)

import cv2,socket,pickle,os
import numpy as np
from RealSense import *
from collections import deque
import numpy as np

rs = RealSense(RS_VGA)		# RS_VGA, RS_720P, or RS_1080P

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,1000000)
server_ip = "10.32.114.243"
server_port = 5555

my_deque = deque(maxlen=100)
maxTime = 0

while True:
    start = time.time()
    (time_, img, depth, accel, gyro) = rs.getData()

    ret , buffer = cv2.imencode(".jpg",img,[int(cv2.IMWRITE_JPEG_QUALITY),30])

    x_as_bytes = pickle.dumps(buffer)

    s.sendto((x_as_bytes),(server_ip,server_port))

    # print(f"Loop Time: {time.time() - start}")
    loopTime = time.time() - start
    my_deque.append(loopTime)
    print(f"Mean: {np.mean(my_deque)}")
    print(f"Max: {np.max(my_deque)}\n")