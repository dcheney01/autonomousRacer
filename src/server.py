
from ultralytics import YOLO
import cv2
import socket
import pickle
import time

model = YOLO('./segmentation/runs/segment/train2/weights/best.pt')

# Set up server socket
s = socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
ip = "10.32.114.243"
port = 5555
s.bind((ip,port))
print("Server is started...")

# Set up PID Controller 
# pid = PID()
# pid.Ki = -.01*0
# pid.Kd = -.01*0
# pid.Kp = -30/250 #degrees per pixel
# frameUpdate = 1
# pid.sample_time = frameUpdate/30.0
# pid.output_limits = (-30,30)
# desXCoord = rgb.shape[0]*3/5
# pid.setpoint = desXCoor

j = 0

while True:
    # Receive message from client
    client_msg, client_addr = s.recvfrom(1000000)
    start = time.time()
    client_ip = client_addr[0]
    img_array = pickle.loads(client_msg)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # possibly unnecessary

    # Get drivable area from image
    result = model.predict(source=img, show=True, verbose=False)

    # Calculate steering angle
    steerCmd = j

    # Send steering angle to client
    s.sendto(str(steerCmd).encode(), client_addr)

    j += 1
    # cv2.imshow('server', img)
    print(f"Loop Time: {time.time()-start}")