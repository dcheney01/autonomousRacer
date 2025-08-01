import cv2
import numpy as np
from typing import Tuple

class CircleOfExpansionEstimator:
    def __init__(self, display=False):
        self.display = display
        self.feature_params = dict( maxCorners = 100,
                                qualityLevel = 0.01,
                                minDistance = 7,
                                )
        self.lk_params = dict( winSize  = (15,15),
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.frames_to_track = 50
        self.old_gray = None
        self.current_features = None
        self.initial_features = None
        self.mask = np.zeros((480, 640, 3), dtype=np.uint8)
        self.x_mask_lim = (0, 640)
        self.y_mask_lim = (0, 150)
        self.tracking_counter = 0

        # self.grid_features = np.array()


    def find_optical_flow_new_points(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        new_features, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.current_features, None, **self.lk_params)
    
        if new_features is not None: # Select features that are in both images
            good_new = new_features[st==1]
            good_old = self.current_features[st==1]

            # ignore features that are outside of the rectangle (0,0) -> (640, 175)
            tmp = good_new.copy()
            good_new = good_new[(tmp[:, 0] > self.x_mask_lim[0]) & (tmp[:, 0] < self.x_mask_lim[1]) & (tmp[:, 1] > self.y_mask_lim[0]) & (tmp[:, 1] < self.y_mask_lim[1])]
            good_old = good_old[(tmp[:, 0] > self.x_mask_lim[0]) & (tmp[:, 0] < self.x_mask_lim[1]) & (tmp[:, 1] > self.y_mask_lim[0]) & (tmp[:, 1] < self.y_mask_lim[1])]
        else:
            return None, None

        if self.display:
            self.mask = np.zeros_like(self.display_img)
            for i, (new, old) in enumerate(zip(good_new, good_old)): # plot keypoints and optical flow vectors
                a, b = new.ravel()
                c, d = old.ravel()

                # slope = (b - d) / (a - c)
                # intercept = b - slope * a
                # y_left = int(slope * 0 + intercept)
                # y_right = int(slope * self.display_img.shape[1] + intercept)

                # if np.abs(slope) > 1:
                # cv2.line(self.mask, (0, y_left), (self.display_img.shape[1], y_right), (0, 255, 0), 2)
                a, b, c, d = int(a), int(b), int(c), int(d)

                self.mask = cv2.line(self.mask, (a, b), (c, d), (0, 255, 0), 2)
                self.display_img = cv2.circle(self.display_img, (a, b), 5, (0, 0, 255), -1)
            self.display_img = cv2.add(self.display_img, self.mask)

        return good_new, good_old

    def find_features_to_track(self, gray:np.ndarray, good_new_features:np.ndarray=None) -> None:
        if good_new_features is None or self.tracking_counter % self.frames_to_track == 0:
            self.current_features = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.initial_features = self.current_features.copy()
            if self.display:
                self.mask = np.zeros_like(self.display_img)
        else:
            self.current_features = good_new_features.reshape(-1, 1, 2)
                
        
    def find_circle_from_optical_flow(self, good_new_features, good_old_features):
        # use least squares
        # vectors = good_new_features - good_old_features
        # A = np.vstack([vectors[:, 1], vectors[:, 0]]).T
        # # b = 0.5 * (np.sum(good_new_features ** 2, axis=1) - np.sum(good_old_features ** 2, axis=1))
        # b = good_old_features[:,0] * A[:,0] - good_old_features[:,1] * A[:,1]

        # center = np.linalg.lstsq(A, b, rcond=None)[0]

        center = np.median(good_new_features, axis=0)

        if np.isnan(center).any():
            return None
        
        if self.display:
            cv2.circle(self.display_img, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
    
        return center

    def get_circle_of_expansion(self, img):
        mask_img = np.zeros_like(img)
        mask_img = cv2.rectangle(mask_img, (self.x_mask_lim[0], self.y_mask_lim[0]), (self.x_mask_lim[1], self.y_mask_lim[1]), (255, 255, 255), -1)

        masked_img = cv2.bitwise_and(img, mask_img)

        if self.display:
            self.display_img = masked_img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        center = None
        good_new_features = None

        if self.old_gray is not None:
            good_new_features, good_old_features = self.find_optical_flow_new_points(gray)

        if good_new_features is not None:
            center = self.find_circle_from_optical_flow(good_new_features, good_old_features)

        self.find_features_to_track(gray, good_new_features) # sets self.current_features

        self.old_gray = gray.copy()
        self.tracking_counter += 1

        if self.display:
            print(center)
            cv2.imshow("img", self.display_img)
            print("\n\n")

        return center

    def get_features_centerpoint(self, img:np.ndarray) -> Tuple[int, int]:
        mask_img = np.zeros_like(img)
        mask_img = cv2.rectangle(mask_img, (self.x_mask_lim[0], self.y_mask_lim[0]), (self.x_mask_lim[1], self.y_mask_lim[1]), (255, 255, 255), -1)
        masked_img = cv2.bitwise_and(img, mask_img)

        if self.display:
            self.display_img = masked_img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

        # only keep features inside the limits
        if features is not None:
            features = features[(features[:, 0, 0] > self.x_mask_lim[0]) & (features[:, 0, 0] < self.x_mask_lim[1]) & (features[:, 0, 1] > self.y_mask_lim[0]) & (features[:, 0, 1] < self.y_mask_lim[1])]
        else:
            return None
        center = np.median(features, axis=0)

        if self.display:
            # draw features on image
            for feature in features:
                x, y = feature.ravel()
                x, y = int(x), int(y)
                cv2.circle(self.display_img, (x, y), 5, (0, 0, 255), -1)
            print(center)
            cv2.imshow("img", self.display_img)
            print("\n\n")
        return center
    

