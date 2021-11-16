###########################################################################
# imports
###########################################################################
import cv2
import numpy as np
from enum import Enum
from typing import Tuple
from abc import ABC


###########################################################################
# FlipType Enum Class
###########################################################################
class FlipType(Enum):
    """Enum class to contain the image flip type enumeration"""
    FLIP_HORIZON = 0
    FLIP_VERTICAL = 1


###########################################################################
# AbstractImageBoundingBoxDataEditor Abstract Class
###########################################################################
class AbstractImageBoundingBoxDataEditor(ABC):
    """An abstract object that contains image and bounding boxes editing tools"""

    def __init__(self):
        super().__init__()

    ####################################
    # Protected
    ####################################

    def _data_translation(self, image_array: np.array, bounding_box_data: dict, x_translation: int,
                          y_translation: int) -> Tuple[dict, np.array]:
        """Translates the given image and its bounding box points
        Args:
            image_array(np.array): image numpy array
            bounding_box_data(dict): contains the bounding box extreme points
            x_translation(int): image offset translation on the x axis
            y_translation(int): image offset translation on the y axis
        Returns:
            dict,np.array: the translated bounding box points and the translated image

        Example:
            bounding_box_data = {"x_min": <int>,"x_max": <int>,"y_min": <int>,"y_max": <int>}

        """
        shift_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
        shifted_image = cv2.warpAffine(image_array, shift_matrix, (image_array.shape[1], image_array.shape[0]))
        shifted_box_data = {"x_min": bounding_box_data["x_min"] + x_translation,
                            "x_max": bounding_box_data["x_max"] + x_translation,
                            "y_min": bounding_box_data["y_min"] + y_translation,
                            "y_max": bounding_box_data["y_max"] + y_translation}
        assert self._is_bounding_box_in_image(image_array, shifted_box_data), f"shifted box data is out of image ->" \
                                                                              f"image shape: {image_array.shape}," \
                                                                              f"shifted box data: {shifted_box_data}"
        return shifted_box_data, shifted_image

    @staticmethod
    def _data_background(image_array: np.array, bounding_box_data: dict, remove_bottom: bool = True):
        """Creates an only background image
        Args:
            image_array(np.array): image numpy array
            bounding_box_data(dict): contains the bounding box extreme points
            remove_bottom(bool): if true the lower part of the image will be cut,else the top part of the image will
                                 be cut
        Returns:
            dict,np.array: the bounding box points (all zero) and the background image

        Example:
            bounding_box_data = {"x_min": 0,"x_max": 0,"y_min": 0,"y_max":0}

        """
        if remove_bottom:
            cropped = image_array[:bounding_box_data['y_min'], :]
        else:
            cropped = image_array[bounding_box_data['y_max']:, :]
        bounding_box_background = {coordinate: 0 for coordinate in bounding_box_data.keys()}
        return bounding_box_background, cv2.resize(cropped, (image_array.shape[1], image_array.shape[0]))

    @staticmethod
    def _data_resize(image_array: np.array, bounding_box_data: dict, new_shape: Tuple[int, int]) -> Tuple[dict,
                                                                                                          np.array]:
        """Reshapes the given image and its bounding box points
        Args:
            image_array(np.array): image numpy array
            bounding_box_data(dict): contains the bounding box extreme points
            new_shape(Tuple): new shape of thee resized image
        Returns:
            dict,np.array: the reshaped bounding box points and the reshaped image

        Example:
            bounding_box_data = {"x_min": <int>,"x_max": <int>,"y_min": <int>,"y_max": <int>}

        """
        image_resized = cv2.resize(image_array, new_shape)
        resized_box_data = {"x_min": round(bounding_box_data["x_min"] * image_resized.shape[1] / image_array.shape[1]),
                            "x_max": round(bounding_box_data["x_max"] * image_resized.shape[1] / image_array.shape[1]),
                            "y_min": round(bounding_box_data["y_min"] * image_resized.shape[0] / image_array.shape[0]),
                            "y_max": round(bounding_box_data["y_max"] * image_resized.shape[0] / image_array.shape[0])}
        return resized_box_data, image_resized

    @staticmethod
    def _data_flip(image_array: np.array, bounding_box_data: dict, flip_type: int) -> Tuple[dict, np.array]:
        """Flips the given image and its bounding box points
        Args:
            image_array(np.array): image numpy array
            bounding_box_data(dict): contains the bounding box extreme points
            flip_type(int): 1 - vertical flip,0 - horizontal
        Returns:
            dict,np.array: the reshaped bounding box points and the reshaped image

        Example:
            bounding_box_data = {"x_min": <int>,"x_max": <int>,"y_min": <int>,"y_max": <int>}

        """
        flipped_image = cv2.flip(image_array, flip_type)
        if flip_type == FlipType.FLIP_VERTICAL.value:
            x_offset = - flipped_image.shape[1]
            y_offset = 0
        elif flip_type == FlipType.FLIP_HORIZON.value:
            x_offset = 0
            y_offset = - flipped_image.shape[0]
        else:
            print("undefined flip type integer: only 1 or 0 is allowed, no alteration to bounding box data!!")
            x_offset = 0
            y_offset = 0
        flipped_box_data = {"x_min": abs(bounding_box_data["x_min"] + x_offset),
                            "x_max": abs(bounding_box_data["x_max"] + x_offset),
                            "y_min": abs(bounding_box_data["y_min"] + y_offset),
                            "y_max": abs(bounding_box_data["y_max"] + y_offset)}
        return flipped_box_data, flipped_image

    @staticmethod
    def _data_normalize(image_array: np.array, bounding_box_data: dict, image_normalization_func: callable):
        """Normalizes the bounding box and the given input image
        Args:
            image_array(np.array): image numpy array
            bounding_box_data(dict): contains the bounding box extreme points
            image_normalization_func(callable): normalization function for the images pixel value
        Returns:
            dict,np.array: the normalized bounding box points and the normalized image

        Example:
            bounding_box_data = {"x_min": <int>,"x_max": <int>,"y_min": <int>,"y_max": <int>}

        """
        normalized_image = image_normalization_func(image_array)
        normalized_box_data = {"x_min": round(bounding_box_data["x_min"] / image_array.shape[1], 3),
                               "x_max": round(bounding_box_data["x_max"] / image_array.shape[1], 3),
                               "y_min": round(bounding_box_data["y_min"] / image_array.shape[0], 3),
                               "y_max": round(bounding_box_data["y_max"] / image_array.shape[0], 3)}
        return normalized_box_data, normalized_image

    @staticmethod
    def _is_bounding_box_in_image(image_array: np.array, bounding_box_data: dict) -> bool:
        """Validates if the given bounding box is in the image's limits
        Args:
            image_array(np.array): image numpy array
            bounding_box_data(dict): contains the bounding box extreme points
        Returns:
            bool: true - bounding box is in the image's limits, else false

        Example:
            bounding_box_data = {"x_min": <int>,"x_max": <int>,"y_min": <int>,"y_max": <int>}

        """
        image_height = image_array.shape[0]
        image_width = image_array.shape[1]
        if bounding_box_data["x_min"] > image_width or bounding_box_data["x_max"] > image_width:
            return False
        if bounding_box_data["y_min"] > image_height or bounding_box_data["y_max"] > image_height:
            return False
        if [value for value in bounding_box_data.values() if value < 0]:
            return False
        return True
