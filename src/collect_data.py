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
import os
import time

enableDepth = False
rs = RealSense(RS_VGA, enableDepth)		# RS_VGA, RS_720P, or RS_1080P

# Use $ ls /dev/tty* to find the serial port connected to Arduino
Car = Arduino("/dev/ttyUSB0", 115200)                # Linux
#Car = Arduino("/dev/tty.usbserial-2140", 115200)    # Mac

time.sleep(3)

Car.zero(1440)      # Set car to go straight.  Change this for your car.
Car.pid(1)          # Use PID control

(time_, rgb, depth, accel, gyro) = rs.getData()

j = 0

test_name = ""
data_folder = f"../data/{test_name}-{time.strftime('%m-%d_%S')}"
rgb_folder = f"{data_folder}/rgb"
gyro_folder = f"{data_folder}/gyro"
accel_folder = f"{data_folder}/accel"

os.makedirs(rgb_folder, exist_ok=True)
os.makedirs(gyro_folder, exist_ok=True)
os.makedirs(accel_folder, exist_ok=True)

print("Starting Data Collection")

Car.drive(1.5)

while(True):
	start = time.time()

	
	(time_, img, _, accel, gyro) = rs.getData()

	# cv2.imwrite(f"{rgb_folder}/img_{j}.jpg", img)
	# np.save(f"{depth_folder}/depth_{j}.npy", depth)
	# np.save(f"{gyro_folder}/accel_{j}.npy", accel)
	# np.save(f"{accel_folder}/gyro_{j}.npy", gyro)
	
	# j += 1

	print(f"Loop: {time.time() - start} s")
