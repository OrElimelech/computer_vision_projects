###########################################################################
# imports
###########################################################################
import os
import cv2
from numpy import array
from typing import Union, Tuple

###########################################################################
# Global Parameters
###########################################################################
ZERO_GAUSSIAN_VARIANCE = 0
CONTOUR_DRAWING_LINE_WIDTH = 6
DRAW_ALL_CONTOURS = -1


###########################################################################
# SegmentedImagesContourHandler Class
###########################################################################


class SegmentedImagesContourHandler:
    """An object that handles with segmented images by finding the segmentation contours and bounding boxes
        Args:
            evidence_output_path(str): evidence output for all the created images, if none will be saved on the current
                                       working environment
    """

    def __init__(self, evidence_output_path: str = "."):
        self.__evidence_output_dir = evidence_output_path
        self.__image_evidence_struct = dict()
        self.__contour_struct = dict()
        self.__bounding_box_struct = list()
        self.__cropped_images = list()
        self.__current_image = None
        self.__original_image = None

    ####################################
    # Public
    ####################################

    @property
    def get_cropped_images(self) -> list:
        """Returns cropped images from the bounding box calculations process.

        Returns:
            list: contains all the cropped images from the bounding box calculations process
         """
        return self.__cropped_images

    @property
    def get_bounding_box_list(self) -> list:
        """Returns the calculated bounding box.

        Returns:
            list: contains all the calculated bounding box arrays
         """
        return self.__bounding_box_struct

    @property
    def get_contours_struct(self) -> dict:
        """Returns the calculated contours.

        Returns:
            dict: contains all the calculated contour arrays
         """
        return self.__contour_struct

    @property
    def get_current_image(self) -> array:
        """Returns the current state image.

        Returns:
            array: current state image
         """
        return self.__current_image

    @get_current_image.setter
    def get_current_image(self, new_image: array) -> None:
        """Sets a new image.
        Args:
            new_image(array): numpy array of an image

         """
        self.__current_image = new_image.copy()
        self.__original_image = new_image.copy()
        self.__image_evidence_struct["original_image"] = self.__original_image

    def create_evidence_images(self) -> None:
        """Creates all evidence images in the given directory"""

        if not os.path.exists(self.__evidence_output_dir):
            os.mkdir(self.__evidence_output_dir)
        for image_name, image_data in self.__image_evidence_struct.items():
            cv2.imwrite(os.path.join(self.__evidence_output_dir, f"{image_name}.jpg"), image_data)

    def calculate_image_contours(self, contour_mode: int = cv2.RETR_EXTERNAL,
                                 contour_method: int = cv2.CHAIN_APPROX_NONE,
                                 sort_mode: Union[str, None] = None, contour_color: Tuple = (255, 0, 0),
                                 gaussian_kernel_size: Tuple = (3, 3)) -> None:
        """Calculates the image's contours.
        Args:
            contour_mode(int): contour mode type (default value = cv2.RETR_EXTERNAL)
            contour_method(int): contour method type (default value = cv2.CHAIN_APPROX_NONE)
            sort_mode(str,None): if None contour swill not be sorted, else the options are: 'LR' - left to right,
                                 'RL' - right to left,'HL' - high to low, 'LH'- low to high
            contour_color(Tuple): color of contour drawing, default value -> (255, 0, 0)
            gaussian_kernel_size(Tuple): size of gaussian blur filter, default value -> (3,3)

         """
        self.__gaussian_blur(gaussian_kernel_size)
        c, h = self.__find_contours(contour_method, sort_mode, contour_mode)
        self.__contour_struct["contours"] = c
        self.__contour_struct["hierarchy"] = h
        self._draw_contours(self.__contour_struct["contours"], contour_color)

    def calculate_contour_bounding_boxes(self, closed_shape: bool = False,
                                         bounding_box_color: Tuple = (0, 255, 0)) -> list:
        """Calculates the bounding box parameters for each founded contour
            Args:
                closed_shape(bool): indicates if the searched shape is close (True) or not (False)
                bounding_box_color(Tuple): bounding box color for drawings

            Returns:
                list: list of found bounded box coordinates
         """
        assert self.__contour_struct, "contour calculation was not done!!"
        bound_rect = [None] * len(self.__contour_struct["contours"])
        contours_poly = [None] * len(self.__contour_struct["contours"])
        for i, contour_data in enumerate(self.__contour_struct["contours"]):
            epsilon = 0.01 * cv2.arcLength(contour_data, closed_shape)
            contours_poly[i] = cv2.approxPolyDP(contour_data, epsilon, closed_shape)
            bound_rect[i] = cv2.boundingRect(contours_poly[i])
            x_range = (int(bound_rect[i][0]), int(bound_rect[i][0] + bound_rect[i][2]))
            y_range = (int(bound_rect[i][1]), int(bound_rect[i][1] + bound_rect[i][3]))
            self.__cropped_images.append(self.__original_image[y_range[0]:y_range[1], x_range[0]:x_range[1]])
        self._draw_bounding_boxes(bound_rect, bounding_box_color)
        self.__bounding_box_struct = bound_rect
        return bound_rect

    ####################################
    # Protected
    ####################################

    def _draw_contours(self, contours: list, contours_color: Tuple, image_input: Union[None, array] = None) -> None:
        """Draws the calculated contours on the original image
            Args:
                contours(list): list of contour structure
                contours_color(Tuple): contour drawing color
                image_input(array/None): if None drawings will be done on the internal image that was set, else
                                         drawings will be done on the input image
         """
        if not image_input:
            draw_contour_image = self.__original_image.copy()
        else:
            draw_contour_image = image_input.copy()
        cv2.drawContours(draw_contour_image, contours, DRAW_ALL_CONTOURS, contours_color, CONTOUR_DRAWING_LINE_WIDTH)
        self.__image_evidence_struct["draw_contour_image"] = draw_contour_image

    def _draw_bounding_boxes(self, bound_rect_list: list, bounding_box_color: Tuple,
                             image_input: Union[None, array] = None):
        """Draws the calculated bounding boxes on the original image
            Args:
                bound_rect_list(list): list of bounding box structures
                bounding_box_color(Tuple): bounding box drawing color
                image_input(array/None): if None drawings will be done on the internal image that was set, else
                                         drawings will be done on the input image
         """
        if not image_input:
            img_with_boxes = self.__original_image.copy()
        else:
            img_with_boxes = image_input.copy()
        for bound_rect in bound_rect_list:
            cv2.rectangle(img_with_boxes, (int(bound_rect[0]), int(bound_rect[1])),
                          (int(bound_rect[0] + bound_rect[2]),
                           int(bound_rect[1] + bound_rect[3])), bounding_box_color)
        self.__image_evidence_struct["bounding_box_image"] = img_with_boxes

    ####################################
    # Private
    ####################################

    @staticmethod
    def __get_center_point_contour(contour: Tuple) -> Tuple:
        """Calculates the contour center point
            Args:
                contour(Tuple): single contour structure


            Returns:
                Tuple: contour center point (X,Y)
         """
        mu = cv2.moments(contour)
        return int(mu['m10'] / mu['m00']), int(mu['m01'] / mu['m00'])

    def __gaussian_blur(self, gaussian_kernel_size: Tuple) -> None:
        """Performs gaussian blur on the current used image
            Args:
                gaussian_kernel_size(Tuple): size of gaussian filter

         """
        assert self.__current_image.any(), "no input image was found!!"
        cv2.GaussianBlur(src=self.__current_image, ksize=gaussian_kernel_size, sigmaX=ZERO_GAUSSIAN_VARIANCE,
                         dst=self.__current_image)
        self.__image_evidence_struct["blurred_image"] = self.__current_image

    def __find_contours(self, contour_method: int, sorted_mode: Union[str, None], contour_mode: int) -> Tuple:
        """Calculating the contours on the current internal image
            Args:
                contour_method(int): type of contour method to be used
                sorted_mode(str/None): mode of sorting method, if None no sorting will be dont
                contour_mode(int): type of contour mode to be used

            Returns:
                Tuple: contours calculation result <contours structure,hierarchy structure>
         """
        self.__current_image = cv2.cvtColor(self.__current_image, cv2.COLOR_BGR2GRAY)
        found_contour, hierarchy = cv2.findContours(self.__current_image, mode=contour_mode,
                                                    method=contour_method)
        if sorted_mode:
            found_contour = self.__sort_contours(sorted_mode, found_contour)
        return found_contour, hierarchy

    def __sort_contours(self, mode: str, contours_struct: Tuple) -> list:
        """Calculating the contours on the current internal image
            Args:
                mode(str): sorting mode - ['LR','RL','HL','LH']
                contours_struct(Tuple): structure of all the calculated contours

            Returns:
                Tuple: contours sorted structure
         """
        mode_struct = {"LR": {"sorting_function": lambda contour: self.__get_center_point_contour(contour)[0],
                              "reverse_flag": False},
                       "RL": {"sorting_function": lambda contour: self.__get_center_point_contour(contour)[0],
                              "reverse_flag": True},
                       "HL": {"sorting_function": lambda contour: self.__get_center_point_contour(contour)[1],
                              "reverse_flag": False},
                       "LH": {"sorting_function": lambda contour: self.__get_center_point_contour(contour)[1],
                              "reverse_flag": True}
                       }
        assert mode in mode_struct, "wrong sorting mode ,the only options available " \
                                    "are: ['LR','RL','HL','LH']"
        return sorted(contours_struct, key=mode_struct[mode]["sorting_function"],
                      reverse=mode_struct[mode]["reverse_flag"])
