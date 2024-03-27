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

	# Use $ ls /dev/tty* to find the serial port connected to Arduino

	# If you get cannot get frame error: use 'sudo pkill -9 python3.6' and wait 15 seconds
**************************************
'''

# import the necessary packages
from Arduino import Arduino
from RealSense import *
import cv2
import pickle
import socket
# from simple_pid import PID

# set up the camera
rs = RealSense(RS_VGA)
(time_, rgb, depth, accel, gyro) = rs.getData()
print("Camera is ready...")

# set up the Car
# Car = Arduino("/dev/ttyUSB0", 115200)                # Linux
# time.sleep(3)
# Car.zero(1440)      # Set car to go straight.  Change this for your car.
# Car.pid(1)          # Use PID control
# print("Car is ready...")

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000)
s.settimeout(0.25)
server_ip = "10.32.114.243"
server_port = 5555
print("Client is ready...")

# Car.drive(3.0)

while(True):
	loopStart = time.time()
	(time_, img, _, accel, gyro) = rs.getData()


	# encode the image
	ret, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
	x_as_bytes = pickle.dumps(buffer)


	# Send image to server
	try:
		s.sendto((x_as_bytes), (server_ip, server_port))
	except:
		print("Timeout sending data to server. Continuing")
		# Car.steer(0.0)
		continue


	# Get steering command from server
	try:
		server_response = s.recvfrom(1000000)
		steer_command = int(server_response[0])
		print(steer_command)
	except:
		print("Timeout waiting for server response. Sending another message")
		# Car.steer(0.0)
		continue


	# Do some steering
	# Car.steer(steer_command)
	

	print(f"Loop Time: {time.time()- loopStart} s")