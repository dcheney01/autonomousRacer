import os
import numpy as np
import cv2
import time
from kachow import CircleOfExpansionEstimator


class PathPlanner:
    def __init__(self, display=True) -> None:
        image = "./00000.png"
        tmp = cv2.imread(image)
        self.test_image = cv2.threshold(tmp, 10, 255, cv2.THRESH_BINARY)[1]
        self.plot_img = tmp.copy()

        self.bottom_of_image = 420

        self.display = display

    def _convolve_rectangle(self, image):
        # Convolve the image with a rectangle filter
        window_width = 20
        stride = 5

        #get sum of pixels in the window
        sums = []
        for i in range(640 // stride):
            sum = np.sum(image[0:self.bottom_of_image,
                               i * stride:i * stride + window_width])
            # print(sum)
            sums.append(sum)

        #get the pixel location of max sum
        max_sum = max(sums)
        max_sum_index = sums.index(max_sum)
        # print(f"max_sum: {max_sum} @ {max_sum_index * stride} px")

        if self.display:
            #draw a vertical line at the max sum index
            cv2.line(self.plot_img,
                     ((max_sum_index * stride) + window_width // 2, 0),
                     ((max_sum_index * stride) + window_width // 2,
                      self.bottom_of_image), (0, 0, 255), 2)
            cv2.imshow("mask", self.plot_img)

        return (max_sum_index * stride) + window_width // 2

    def _crop_bottom_rectangle(self, image):
        cropped_img = image[:self.bottom_of_image, :]
        # if self.display:
        #     cv2.imshow("Cropped Image", cropped_img)
        return cropped_img

    # def _get_centroid_of_mask(self, mask):
    #     # Get the centroid of the mask
    #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #     mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]
    #     contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
    #                                    cv2.CHAIN_APPROX_SIMPLE)
    #     if len(contours) == 0:
    #         return None
    #     cnt = contours[0]
    #     M = cv2.moments(cnt)
    #     if M["m00"] == 0:
    #         return None
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #     return cX, cY

    def _grid_rectangle(self, image):
        # Grid the image into rectangles
        num_rectangles = 13
        window_width = 640 // num_rectangles

        #get sums of pixels in each rectangle
        sums = []
        for i in range(num_rectangles):
            sum = np.sum(image[0:self.bottom_of_image,
                               i * window_width:(i + 1) * window_width])
            sums.append(sum)

        max_sum = max(sums)
        max_sum_index = sums.index(max_sum)

        if self.display:
            for i in range(num_rectangles):
                cv2.line(image, (int(i * window_width), 0),
                         (int(i * window_width), self.bottom_of_image),
                         (0, 255, 255), 2)
        if self.display:
            cv2.imshow("Grid", image)

        return max_sum_index * window_width + window_width // 2

    def _get_road_boundary_lines(self, image):
        # Get the road boundary lines
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        if self.display:
            cv2.imshow("Edges", edges)

        # Apply HoughLines
        lines = cv2.HoughLines(edges, 5, np.radians(5), 100)

        self._filter_classify_lines(image, lines)

        return lines

    def _filter_classify_lines(self, image, lines):
        #classify lines into left and right by positive and negative slope
        left_lines = []
        right_lines = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                try:
                    slope = (y2 - y1) / (x2 - x1)
                    if slope > 0:
                        right_lines.append((x1, y1, x2, y2))
                    else:
                        left_lines.append((x1, y1, x2, y2))
                except ZeroDivisionError:
                    continue

            if len(right_lines) > 0:
                right_avg = np.median(right_lines, axis=0)
                # if self.display:
                #     cv2.line(self.plot_img,
                #              (int(right_avg[0]), int(right_avg[1])),
                #              (int(right_avg[2]), int(right_avg[3])),
                #              (255, 0, 255), 2)

            if len(left_lines) > 0:
                left_avg = np.median(left_lines, axis=0)
                # if self.display:
                #     cv2.line(self.plot_img,
                #              (int(left_avg[0]), int(left_avg[1])),
                #              (int(left_avg[2]), int(left_avg[3])), (255, 0, 0),
                #              2)

    def get_line_to_follow(self, image):
        self.plot_img = image.copy()
        cropped_img = self._crop_bottom_rectangle(image)
        self._get_road_boundary_lines(cropped_img)
        grid_x = self._grid_rectangle(cropped_img)
        convolution_x = self._convolve_rectangle(cropped_img)
        # centroid = self._get_centroid_of_mask(cropped_img)

        return grid_x, convolution_x


if __name__ == "__main__":

    masks = os.listdir("../lap1/SegmentationClass/")
    masks = sorted(masks)

    images = os.listdir("../first_lap/full_lap1_firstCorner/rgb/")
    images = sorted(images)

    est = CircleOfExpansionEstimator()

    for mask, image in zip(masks, images):
        mask_img = cv2.imread(f"../lap1/SegmentationClass/{mask}")
        rgb_img = cv2.imread(f"../first_lap/full_lap1_firstCorner/rgb/{image}")
        path_planner = PathPlanner()
        grid_x, conv_x = path_planner.get_line_to_follow(mask_img)
        cv2.line(rgb_img, (grid_x, 0), (grid_x, 420), (0, 255, 0), 2)
        cv2.line(rgb_img, (conv_x, 0), (conv_x, 420), (0, 0, 255), 2)
        center = est.get_features_centerpoint(rgb_img)
        print(center)
        feature_x = int(center[0,0])
        cv2.line(rgb_img, (feature_x, 0), (feature_x, 420), (255, 0, 0), 2)

        #add text along each lines
        cv2.putText(rgb_img, "Grid", (grid_x, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
        cv2.putText(rgb_img, "Convolution", (conv_x, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(rgb_img, "Feature", (feature_x, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("RGB", rgb_img)
        key = cv2.waitKey(1)
        if key == 'q':
            cv2.destroyAllWindows()
