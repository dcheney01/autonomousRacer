# python3 ECEnRacer.py
''' 
This program is for ECEN-631 BYU Race
*************** RealSense Package ***************
From the Realsense camera:
	RGB Data
	Depth Data
	Gyroscope Data
	Accelerometer Data
*************** Arduino Package ****************
	Steer(int degree) : -30 (left) to +30 (right) degrees
	Drive(float speed) : -3.0 to 3.0 meters/second
	Zero(int PWM) : Sets front wheels going straight around 1500
	Encoder() : Returns current encoder count.  Reset to zero when stop
	Pid(int flag) : 0 to disable PID control, 1 to enable PID control
	KP(float p) : Proporation control 0 ~ 1.0 : how fast to reach the desired speed.
	KD(float d) : How smoothly to reach the desired speed.

	EXTREMELY IMPORTANT: Read the user manual carefully before operate the car

	# If you get cannot get frame error: use 'sudo pkill -9 python3.6' and wait 15 seconds
**************************************
'''

# import the necessary packages
from Arduino import Arduino
from RealSense import *
import cv2
import imagezmq
import zmq
import threading
# from simple_pid import PID

enableDepth = False
rs = RealSense(RS_VGA, enableDepth)		# RS_VGA, RS_720P, or RS_1080P

# Use $ ls /dev/tty* to find the serial port connected to Arduino
# Car = Arduino("/dev/ttyUSB0", 115200)                # Linux
#Car = Arduino("/dev/tty.usbserial-2140", 115200)    # Mac
time.sleep(3)

# Car.zero(1440)      # Set car to go straight.  Change this for your car.
# Car.pid(1)          # Use PID control

(time_, rgb, depth, accel, gyro) = rs.getData()
def sender_start(connect_to=None):
    sender = imagezmq.ImageSender(connect_to=connect_to)
    # sender.zmq_socket.setsockopt(zmq.LINGER, 0)  # prevents ZMQ hang on exit
    # # NOTE: because of the way PyZMQ and imageZMQ are implemented, the
    # #       timeout values specified must be integer constants, not variables.
    # #       The timeout value is in milliseconds, e.g., 2000 = 2 seconds.
    # sender.zmq_socket.setsockopt(zmq.RCVTIMEO, 2000)  # set a receive timeout
    # sender.zmq_socket.setsockopt(zmq.SNDTIMEO, 2000)  # set a send timeout
    return sender

server_address = 'tcp://10.32.114.243:5555'
sender = sender_start(server_address)
host_name = "LightningMcQueen"
print("Started Client")

# ## SETUP PID Controller
# pid = PID()
# pid.Ki = -.01*0
# pid.Kd = -.01*0
# pid.Kp = -30/250 #degrees per pixel
# frameUpdate = 1
# pid.sample_time = frameUpdate/30.0
# pid.output_limits = (-30,30)
# desXCoord = rgb.shape[0]*3/5
# pid.setpoint = desXCoord

# i = 1
# angle = 0
# FAST_SPEED = 1.3
# SLOW_SPEED = 0.5
# speed = FAST_SPEED
# Car.drive(FAST_SPEED)


# # You can use kd and kp commands to change KP and KD values.  Default values are good.
# # loop over frames from Realsense
j = 0


start = time.time()
while(True):
	loopStart = time.time()
	(time_, img, _, accel, gyro) = rs.getData()

	smaller_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

	if time.time() - start > 0.1:
		# try:
		reply_from_server = sender.send_image(host_name, smaller_img)
		# except (zmq.ZMQError, zmq.ContextTerminated, zmq.Again):
			# startClose = time.time()
			# if 'sender' in locals():
			# 	print('Closing ImageSender.')
			# 	sender.close()
			# print('Restarting ImageSender.')
			# sender = sender_start(server_address)
			# print(f"Reconnect time: {time.time() - startClose} s\n\n")
		start = time.time()


		# print(reply_from_server)
	print(f"Loop Time: {time.time()- loopStart} s")

