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


#TURN LOGIC
turn_order = [0,1,2,3]
corner_thresholds = [(0.01, .03), (0.01, .03), (0.01, .03), (0.01, .03)]
corners_thresholds = [corner_thresholds[i] for i in turn_order]
TURN_COUNTER = 0
TURNING = False


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

# s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000)
# s.settimeout(0.25)
# server_ip = "10.32.114.243"
# server_port = 5555
# print("Client is ready...")

def get_blue_mask(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# threshold on hsv
	lower = (75, 150, 0)
	upper = (110, 255, 255)
	mask = cv2.inRange(hsv, lower, upper)
	return mask

def depth_x(depth_img):
	setpoint_y_line = 75
	x_coord = np.argmin(depth_img[setpoint_y_line,:])
	return x_coord


def we_need_to_turn(blue_mask):
	global TURN_COUNTER, TURNING
	blues = np.sum(blue_mask)/(640*480*255)

	low, high = corner_thresholds[TURN_COUNTER]

	#we need to turn if blue mask is over a threshold
	#we do not need to turn if blue mask is under a threshold. 
	if blues > high:
		TURNING = True
		return True
	elif blues < low:
		if TURNING:
			TURN_COUNTER += 1
		TURNING = False
		return False
	else:
		#return whatever we were doing last
		return TURNING



Car.drive(2.5)

while(True):
	loopStart = time.time()

	rgb, depth = rs.getData(enable_depth=True, enable_rgb=True)

	blue_mask = get_blue_mask(rgb)

	if we_need_to_turn(blue_mask):
		#turn logic
		steer_command = -15
	else:
		cur_x = depth_x(depth)

		steer_command = pid(cur_x)

	# Do some steering
	Car.steer(steer_command)

	print(f"Loop Time: {time.time()- loopStart} s")




	########## THIS IS ALL OF THE SOCKET STUFF ###################
	# # encode the image
	# # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# img = cv2.resize(img, (160, 120))
	# # print(img.shape)
	# ret, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
	# x_as_bytes = pickle.dumps(buffer)
	# print(x_as_bytes.__sizeof__())

	# # Send image to server
	# try:
	# 	# print(x_as_bytes.__sizeof__())
	# 	# s.sendto((x_as_bytes), (server_ip, server_port))
	# 	s.sendall(x_as_bytes)
	# except:
	# 	print("Timeout sending data to server. Continuing")
	# 	Car.drive(0.)
	# 	# cv2.imwrite('depth.jpg', depth)
	# 	# Car.steer(0.0)
	# 	# Car.drive(1.5)
	# 	continue


	# # Get steering command from server
	# try:
	# 	server_response = s.recvfrom(receive_buffer_size)
	# 	steer_command = int(server_response[0])
	# 	Car.drive(2.5)
	# 	print(steer_command)
	# except Exception as e:
	# 	# print(f"Timeout waiting for server response. Sending another message. Got {e}")

	# 	start = time.time()

	# 	#get steering angle from depth instead
	# 	# depth = rs.getData(enable_depth=True, enable_rgb=False)
	# 	Car.drive(0.)

	# 	print(f"Timeout. SLOWING DOWN. Time to get depth: {time.time() - start}")
		
	# 	continue


	

