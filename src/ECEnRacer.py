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
from simple_pid import PID

from kachow import CircleOfExpansionEstimator
coeEstimator = CircleOfExpansionEstimator(display=False)

# Set up PID Controller 
pid = PID()
pid.Ki = 0.0
pid.Kd = 0.0
pid.Kp = -40/320 #degrees per pixel
frameUpdate = 1
pid.sample_time = frameUpdate/30.0
pid.output_limits = (-8,8)
desXCoord = 640 * 1/3
pid.setpoint = desXCoord


# set up the camera
rs = RealSense(RS_VGA, enable_depth=True)
rgb = rs.getData()
print("Camera is ready...")

# set up the Car
Car = Arduino("/dev/ttyUSB0", 115200)                # Linux
time.sleep(3)
Car.zero(1400)      # Set car to go straight.  Change this for your car.
Car.pid(1)          # Use PID control
print("Car is ready...")

# coeEstimator = CircleOfExpansionEstimator(display=True)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_buffer_size = 72
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 3000)
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer_size)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setblocking(False)

s.settimeout(0.15)
server_ip = "10.37.85.80"
# server_ip = "10.32.114.243"
server_port = 5555
s.connect((server_ip, server_port))
print("Client is connected to server...")

time.sleep(2.0)

Car.drive(2.5)

while(True):
	loopStart = time.time()
	img = rs.getData()


	# encode the image
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (160, 120))
	# print(img.shape)
	ret, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
	x_as_bytes = pickle.dumps(buffer)
	print(x_as_bytes.__sizeof__())

	# Send image to server
	try:
		# print(x_as_bytes.__sizeof__())
		# s.sendto((x_as_bytes), (server_ip, server_port))
		s.sendall(x_as_bytes)
	except:
		print("Timeout sending data to server. Continuing")
		Car.drive(0.)
		# cv2.imwrite('depth.jpg', depth)
		# Car.steer(0.0)
		# Car.drive(1.5)
		continue


	# Get steering command from server
	try:
		server_response = s.recvfrom(receive_buffer_size)
		steer_command = int(server_response[0])
		Car.drive(2.5)
		print(steer_command)
	except Exception as e:
		# print(f"Timeout waiting for server response. Sending another message. Got {e}")

		start = time.time()

		#get steering angle from depth instead
		# depth = rs.getData(enable_depth=True, enable_rgb=False)
		Car.drive(0.)

		print(f"Timeout. SLOWING DOWN. Time to get depth: {time.time() - start}")
		
		continue


	# Do some steering
	Car.steer(steer_command)
	

	print(f"Loop Time: {time.time()- loopStart} s")
