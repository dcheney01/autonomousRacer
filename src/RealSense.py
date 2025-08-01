# Camera.py
'''
******* Realsense camera as the sensor ***************
The Intel Realsense 435i camera provides
    RGB Data
    Depth Data
	Gyroscope Data
	Accelerometer Data
***********************************************
'''
# import the necessary packages
import pyrealsense2 as rs
import cv2
import numpy as np
import time

RS_VGA = 0
RS_720P = 1
RS_1080P = 2
class RealSense:
    def __init__(self, Resolution, enable_depth=False):
         # configure rgb, depth, gyro, accel streams
        if Resolution == RS_720P:
            rgbSize = [1280, 720]
            depthSize = [1280, 720]
        elif Resolution == RS_1080P:
            rgbSize = [1920, 1080]
            depthSize = [1280, 720]     # depth camera only allows upto 720P
        else:
            rgbSize = [640, 480]
            depthSize = [640, 480]
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, rgbSize[0], rgbSize[1], rs.format.bgr8, 60)
        config.enable_stream(rs.stream.depth, depthSize[0], depthSize[1], rs.format.z16, 60)  
        # if enable_depth:
        # self.enable_depth = enable_depth
        # config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
        # config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

        # Start streaming
        prof = self.pipeline.start(config)
        s = prof.get_device().query_sensors()[1]
        s.set_option(rs.option.exposure, 125)

        self.colorizer = rs.colorizer()
        # Create alignment primitive with color as its target stream:
        self.align = rs.align(rs.stream.color)

    def __del__(self):
        self.pipeline.stop()

    # Functions for accessing IMU data
    def gyro_data(self, gyro):
        return np.asarray([gyro.x, gyro.y, gyro.z])

    def accel_data(self, accel):
        return np.asarray([accel.x, accel.y, accel.z])

    def getData(self, enable_depth=True, enable_rgb=True):
        # start realsense pipeline
        rsframes = self.pipeline.wait_for_frames()
        if enable_rgb:
            color_frame = rsframes.get_color_frame()
            rgb = np.asanyarray(color_frame.get_data())
            # frame = rgb
        # iterate through camera/IMU data, updating global variable
        for rsframe in rsframes:
            # Retrieve IMU data
            if rsframe.is_motion_frame():
                pass
                # accel = self.accel_data(rsframe.as_motion_frame().get_motion_data())
                # gyro = self.gyro_data(rsframe.as_motion_frame().get_motion_data())
            # Retrieve depth data
            if rsframe.is_depth_frame():
                if enable_depth:
                    rsframes = self.align.process(rsframes)
                    # Update color and depth frames:ss
                    depth_frame = rsframes.get_depth_frame()
                    # Convert to numpy array
                    depth = cv2.normalize(~np.asanyarray(depth_frame.get_data()), None, 255, 0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    # frame = depth
                    # depth = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        return rgb, depth
        # if not enable_depth:
        #     return (time.time(), rgb, None, None, None)
        # # return(time.time(), rgb, depth, accel, gyro)
        # return(time.time(), rgb, depth, None, None)



