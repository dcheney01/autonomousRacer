# run this program on each RPi to send a labelled image stream
import socket
import time
from imutils.video import VideoStream
import imagezmq
from RealSense import *

sender = imagezmq.ImageSender(connect_to='tcp://10.32.114.243:5555')
rs = RealSense(RS_VGA)		# RS_VGA, RS_720P, or RS_1080P
(time_, rgb, depth, accel, gyro) = rs.getData()

host_name = socket.gethostname() # send RPi hostname with each image

time.sleep(2.0)  # allow camera sensor to warm up

while True:  # send images as stream until Ctrl-C
    (time_, img, depth, accel, gyro) = rs.getData()
    reply_from_server = sender.send_image(host_name, img)
    print(reply_from_server)