import os
import numpy as np
import cv2
import time


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
            cv2.imshow("Path", self.plot_img)

        return (max_sum_index * stride) + window_width // 2

    def _crop_bottom_rectangle(self, image):
        cropped_img = image[:self.bottom_of_image, :]
        # if self.display:
        #     cv2.imshow("Cropped Image", cropped_img)
        return cropped_img

    def get_line_to_follow(self, image):
        self.plot_img = image.copy()
        cropped_img = self._crop_bottom_rectangle(image)
        return self._convolve_rectangle(cropped_img)


if __name__ == "__main__":

    masks = os.listdir("../lap1/SegmentationClass/")
    masks = sorted(masks)

    images = os.listdir("../first_lap/full_lap1_firstCorner/rgb/")
    images = sorted(images)

    for mask, image in zip(masks, images):
        mask_img = cv2.imread(f"../lap1/SegmentationClass/{mask}")
        rgb_img = cv2.imread(f"../first_lap/full_lap1_firstCorner/rgb/{image}")
        path_planner = PathPlanner()
        x_coord = path_planner.get_line_to_follow(mask_img)
        cv2.line(rgb_img, (x_coord, 0), (x_coord, 480), (0, 0, 255), 2)
        cv2.imshow("Path", rgb_img)
        key = cv2.waitKey(0)

        if key == 'q':
            cv2.destroyAllWindows()
