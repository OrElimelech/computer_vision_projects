###########################################################################
# imports
###########################################################################
import cv2
import queue
import numpy as np
from typing import Tuple

###########################################################################
# Global Parameters
###########################################################################

RESET_VALUE = 32
ESC_VALUE = 27
LINE_SIZE = 3
IMG_SHAPE = (200, 700, 3)
WHITE_COLOR = (255, 255, 255)
WINDOW_TITLE = "Drawer Window"
QUEUE_CELL_NUM = 2


###########################################################################
# Drawer Class
###########################################################################

class Drawer:
    """An object that enables image drawing
        Args:
            line_size(int): line width
            image_shape(Tuple): shape of the image -> (cols,rows,channel number)
            line_color(Tuple): color of the line -> (B-><0,255>,G-><0,255>,R-><0,255>)
    """

    def __init__(self, line_size: int = LINE_SIZE, image_shape: Tuple = IMG_SHAPE, line_color: Tuple = WHITE_COLOR):
        self.__image_conf = self.__create_image_config_struct(line_size, image_shape, line_color)
        self.__writer_image = self.__build_writer_image()
        self.__point_q = queue.Queue(maxsize=2)
        self.__draw_flag = False
        self.__run_drawer()

    ####################################
    # Public
    ####################################

    def is_image_empty(self) -> bool:
        """Validates if anything was written in the image.

        Returns:
            bool: true - something was changed in the image, else False
         """

        if self.__writer_image.any() != self.__build_writer_image().any():
            return False
        else:
            return True

    @property
    def get_current_image(self) -> np.array:
        """Returns the most recent image that was saved.

        Returns:
            np.array: the most recent image that was saved
         """
        return self.__writer_image

    ####################################
    # Private
    ####################################

    @staticmethod
    def __create_image_config_struct(line_size: int, image_shape: Tuple, line_color: Tuple) -> dict:
        """Creates an image configuration structure
            Args:
                line_size(int): line width
                image_shape(Tuple): shape of the image -> (cols,rows,channel number)
                line_color(Tuple): color of the line -> (B-><0,255>,G-><0,255>,R-><0,255>)


            Returns:
                dict: image configuration structure
         """

        return {
            "line_size": line_size,
            "image_shape": image_shape,
            "line_color": line_color}

    def __build_writer_image(self) -> np.array:
        """Returns an empty black image.

        Returns:
            np.array: an empty black image
         """
        return np.zeros(self.__image_conf["image_shape"], np.uint8)

    def __click_and_draw(self, event: int, x: int, y: int, flags=None, param=None) -> None:
        """Creates an image configuration structure
            Args:
                event(int): event integer type for this callback function
                x(int): x value of the pixel point of the mouse location in the image
                y(int): y value of the pixel point of the mouse location in the image
                flags(None): irrelevant, only used as a placeholder for callback setter function
                param(None): irrelevant, only used as a placeholder for callback setter function

         """

        if event == cv2.EVENT_LBUTTONDOWN:
            self.__draw_flag = True
            if not self.__point_q.empty():
                self.__point_q.get()
            self.__point_q.put((x, y))
            cv2.line(self.__writer_image, (x, y), (x, y), self.__image_conf["line_color"],
                     self.__image_conf["line_size"])
            cv2.imshow(WINDOW_TITLE, self.__writer_image)
        if event == cv2.EVENT_LBUTTONUP:
            self.__draw_flag = False
        if event == cv2.EVENT_MOUSEMOVE:
            if self.__draw_flag:
                cv2.line(self.__writer_image, self.__point_q.get(), (x, y), self.__image_conf["line_color"],
                         self.__image_conf["line_size"])
                self.__point_q.put((x, y))
                cv2.imshow(WINDOW_TITLE, self.__writer_image)

    def __run_drawer(self) -> None:
        """Runs the drawer activation process"""
        while True:
            key = cv2.waitKey(1)
            cv2.namedWindow(WINDOW_TITLE)
            cv2.setMouseCallback(WINDOW_TITLE, self.__click_and_draw)
            cv2.imshow(WINDOW_TITLE, self.__writer_image)

            if key == ESC_VALUE:
                cv2.destroyAllWindows()
                break
            if key == RESET_VALUE:
                self.__writer_image = self.__build_writer_image()
                self.__point_q = queue.Queue(maxsize=QUEUE_CELL_NUM)
