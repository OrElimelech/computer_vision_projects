###########################################################################
# imports
###########################################################################
import cv2
import numpy as np
from functools import partial
from typing import Union, Iterator, Tuple
from utilities.xml_car_plate_data_generator import XmlCarPlateDataGenerator
from lib.images_editor.abstract_image_bounding_box_data_editor import AbstractImageBoundingBoxDataEditor, FlipType


###########################################################################
# ImageCarPlateDataGenerator
###########################################################################
class ImageCarPlateDataGenerator(AbstractImageBoundingBoxDataEditor):
    """An object that generates training data according to the initialized properties - images and target vector
        Args:
            images_path(str): path to images location
            xml_path(str): path to the dta set annotation files (xml)
            resized_shape(tuple): size of the image to reshaped -> (height,width)
            normalization_func(func): normalization function to alter image pixel values
            add_vertical_flip(bool): add an image with vertical flip
            only_background(bool): add an only background image out of the original image
            crop_percentage_index(None/float): if none or bigger than 1 the whole list of found xml will be iterated, e
                                              else will start cropping from the calculated index
            crop_from_start_flag(bool): true - indicates that the cropping will be done from the start of the found xml
                                        list, else from the en
    """

    def __init__(self,
                 images_path: str, xml_path: str, resized_shape: tuple,
                 normalization_func: callable, add_vertical_flip: bool = False, only_background: bool = False,
                 crop_percentage_index: Union[float, None] = None, crop_from_start_flag: bool = True):
        super().__init__()
        self.__normalization_func = normalization_func
        self.__add_vertical_flip = add_vertical_flip
        self.__resized_shape = resized_shape
        self.__only_background = only_background
        self.__xml_data_parser = XmlCarPlateDataGenerator(xml_path, images_path,
                                                          crop_percentage_index, crop_from_start_flag)
        self.__data_generator = self.__image_bounding_box_generator()
        self.__max_generated_images = self.__xml_data_parser.get_xml_files_num()
        self.__batch_size = 1 + [self.__add_vertical_flip, self.__only_background].count(True)

    def __iter__(self) -> Iterator:
        """iterator method"""
        return self.__data_generator

    def __next__(self) -> tuple:
        """Builds the next value of the generator
        Returns:
            tuple: structure of the generator value -> image of numpy array, target vector
        """
        return next(self.__data_generator)

    ####################################
    # Public
    ####################################

    @property
    def get_max_generated_images(self) -> int:
        """Returns the maximum generated image
        Returns:
            int:  number of steps per epoch
        """
        return self.__max_generated_images

    @property
    def get_batch_size(self) -> int:
        """Returns the maximum batch size
        Returns:
            int:  maximum batch size
        """
        return self.__batch_size

    ####################################
    # Private
    ####################################

    def __image_bounding_box_generator(self) -> Tuple[np.array, np.array]:
        """Builds the generator structure
        Returns:
            Union[np.array, np.array]: generator structure  -> first index : image array, second index : target vector
        """
        while True:
            for xml_data in self.__xml_data_parser:
                images, targets = [], []
                bounding_box = xml_data[0]
                image_path = xml_data[1]
                original_image = cv2.imread(image_path)
                bounding_box_resized, resized_image = self._data_resize(original_image, bounding_box,
                                                                        self.__resized_shape)
                image_array_norm, target_vector = self._data_editor("resize_image", original_image, bounding_box,
                                                                    True)
                targets.append(target_vector), images.append(image_array_norm)
                if self.__add_vertical_flip:
                    image_array_norm, target_vector = self._data_editor("flip_image", resized_image,
                                                                        bounding_box_resized,
                                                                        True)
                    targets.append(target_vector), images.append(image_array_norm)
                if self.__only_background:
                    image_array_norm, target_vector = self._data_editor("create_background", resized_image,
                                                                        bounding_box_resized,
                                                                        False)
                    targets.append(target_vector), images.append(image_array_norm)
                yield np.asarray(images), np.asarray(targets)

    def _data_editor(self, editing_type: str, image_array: np.array, bounding_box_struct: dict,
                     found_object_flag: bool) -> Tuple[np.array, np.array]:
        """Edits the data and prepare it for training
        Args:
            editing_type(str): type of action , currently supported -> "create_background","flip_image","resize_image"
            image_array(np.array): numpy image array
            bounding_box_struct(dict): bounding box image data
            found_object_flag(bool): true -> image as an object, else -> false
        Returns:
            Tuple[np.array, np.array]y: normalized image and a normalized target vector
         """
        editing_operation_struct = {
            "create_background": partial(self._data_background, image_array, bounding_box_struct),
            "flip_image": partial(self._data_flip, image_array, bounding_box_struct, FlipType.FLIP_HORIZON.value),
            "resize_image": partial(self._data_resize, image_array, bounding_box_struct, self.__resized_shape)
        }
        try:
            bounding_box_new, image_new = editing_operation_struct[editing_type]()
        except KeyError:
            assert False, f"wrong editing type action, those are the only_possibilities: " \
                          f"{list(editing_operation_struct.keys())}"
        bounding_box_norm, image_norm = self._data_normalize(image_new, bounding_box_new, self.__normalization_func)
        target_vector = [int(found_object_flag)] + [coordinate for coordinate in bounding_box_norm.values()]
        return image_norm, target_vector
