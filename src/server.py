
from ultralytics import YOLO
import cv2
import socket
import pickle
import time
from simple_pid import PID
from path_planner import PathPlanner
import numpy as np

model = YOLO('./segmentation/runs/segment/train2/weights/best.pt')

# Set up server socket
print("Listenting for client...")
s = socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
receive_buffer_size = 3000
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 72)
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, receive_buffer_size)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setblocking(False)
s.settimeout(0.1)
# ip = "10.32.114.243"
ip = "10.37.102.0"
port = 5555
s.bind((ip,port))
print(f"Server is ready...")

# Set up PID Controller 
pid = PID()
pid.Ki = 0.0
pid.Kd = 0.000
pid.Kp = -20/320 #degrees per pixel
frameUpdate = 1
pid.sample_time = frameUpdate/30.0
pid.output_limits = (-10,10)
desXCoord = 640 * 0.5
pid.setpoint = desXCoord
path_planner = PathPlanner(display=False)
print("PID Controller is ready...")

while True:
    try:
        # Receive message from client
        client_msg, client_addr = s.recvfrom(receive_buffer_size)
        start = time.time()
        client_ip = client_addr[0]
        img_array = pickle.loads(client_msg)
        print("Received image from client...")
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # possibly unnecessary
        img = cv2.resize(img, (640, 480))

        # Get drivable area from image
        result = model.predict(source=img, show=False, verbose=False)
        try:
            mask_img = result[0].masks.data[0].cpu().numpy()
            if mask_img is None:
                raise Exception("No mask image detected")
        except Exception as e:
            print(e)
            s.sendto("0".encode(), client_addr)
            continue

        # Calculate steering angle
        curr_x_coord = path_planner.get_line_to_follow(mask_img)
        steerCmd = int(pid(curr_x_coord))
        print(f"Current X Coord: {curr_x_coord}")
        print(f"Steering Command: {steerCmd}\n")

        # Send steering angle to client\
        # print((str(steerCmd).encode()).__sizeof__())
        s.sendto(str(steerCmd).encode(), client_addr)

        # plot masked image with steering angle
        mask = cv2.resize(mask_img.astype(np.uint8)*255, (640, 480))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        opacity = 0.3
        cv2.addWeighted(mask, opacity, img, 1 - opacity, 0, img)
        cv2.line(img, (curr_x_coord, 0), (curr_x_coord, 480), (0, 0, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

        print(f"Loop Time: {time.time()-start}")
    except Exception as e:
        print(f"No data in buffer. Got {e}")
        continue