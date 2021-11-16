###########################################################################
# imports
###########################################################################
import os
import re
import cv2
import numpy as np
from typing import Union, Iterator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


###########################################################################
# ImageGeneratorHandler Class
###########################################################################


class ImageGeneratorHandler:
    """An object that handles the creation of image generators for training processes
        Args:
            image_generators_struct(dict): structure that contains all the instructions for generator creations

        Example: image_generators_struct = {generator1 : {"zca_epsilon": 1e-6,"width_shift_range": 0.4},
                                            generator2 : {"zca_epsilon": 1e-6,"width_shift_range": 0.7}}
    """

    def __init__(self, image_generators_struct: Union[dict, None] = None):
        self.__created_image_generators = self.__create_configured_generators(image_generators_struct)

    ####################################
    # Public
    ####################################

    def create_generator_from_image_set(self, generator_name: str, directory: str, image_type: str, image_size: tuple,
                                        class_id: int,
                                        conf_struct: Union[dict, None] = None,
                                        integer_sorting: bool = False) -> ImageDataGenerator:
        """Creates a generator from images inside a single directory
              Args:
                  generator_name(str): name of the initialized created generator
                  directory(str): the image's folder directory
                  image_type(str): image type -> "jpg","png"....
                  image_size(tuple): size of the image, if not same as the image's original size the image's size will
                                     be altered
                  class_id(int): id number to be as a label for the images in that class
                  conf_struct(dict): a structure of parameters to initialize the data generator
                  integer_sorting(bool): if true: sort the order of the image generator by an integer that is part of
                                         the image's name -> "0.jpg","1.jpg"..

              Returns:
                  ImageDataGenerator: the created image data generator
           """
        images_arrays = self._get_images_arrays(directory, image_type, image_size, integer_sorting)
        required_generator = self._get_created_generator(generator_name)
        if conf_struct is None:
            conf_struct = dict()
        conf_struct["x"] = images_arrays
        conf_struct["y"] = np.asarray([class_id] * images_arrays.shape[0])
        return required_generator.flow(**conf_struct)

    def create_generator_from_flow(self, generator_name: str, x_label: np.array, y_label: Union[None, np.array] = None,
                                   conf_struct: Union[dict, None] = None) -> Iterator:
        """Creates a generator from numpy array
              Args:
                  generator_name(str): name of the initialized created generator
                  x_label(np.array): numpy array of images
                  y_label(None, np.array): targets numpy array, if None no targets will be created in the generator
                  conf_struct(dict): a structure of parameters to initialize the data generator


              Returns:
                  Iterator: the created image data generator
           """
        required_generator = self._get_created_generator(generator_name)
        if conf_struct is None:
            conf_struct = dict()
        conf_struct["x"], conf_struct["y"] = x_label, y_label
        return required_generator.flow(**conf_struct)

    def create_generator_for_directory(self, generator_name: str, directory_path: str,
                                       conf_struct: Union[None, dict] = None) -> Iterator:
        """Creates a generator from numpy array
              Args:
                  generator_name(str): name of the initialized created generator
                  directory_path(str): directory with the following structure: root folder -
                                                                                           - class_0_folder ->images
                                                                                           - class_1_folder ->images
                  conf_struct(dict): a structure of parameters to initialize the data generator


              Returns:
                  Iterator: the created image data generator
           """
        required_generator = self._get_created_generator(generator_name)
        if conf_struct is None:
            conf_struct = dict()
        conf_struct["directory"] = directory_path
        return required_generator.flow_from_directory(**conf_struct)

    ####################################
    # Protected
    ####################################

    def _get_created_generator(self, generator_name: str) -> ImageDataGenerator:
        """Retrieves already initialized base image data generator
              Args:
                  generator_name(str): name of the initialized created generator

              Returns:
                  ImageDataGenerator: the retrieved image data generator
           """
        assert self.__created_image_generators, "Generator instructions was not given as an initializer member"
        assert generator_name in self.__created_image_generators, f"wrong generator name, " \
                                                                  f"the only options are: " \
                                                                  f"{list(self.__created_image_generators.keys())}"
        return self.__created_image_generators[generator_name]

    def _get_images_arrays(self, directory: str, image_type: str, image_size: tuple, integer_sorting: bool = False) \
            -> np.array:
        """Creates an images numpy array from a given directory
              Args:
                  directory(str): the image's folder directory
                  image_type(str): image type -> "jpg","png"....
                  image_size(tuple): size of the image, if not same as the image's original size the image's size will
                                     be altered
                  integer_sorting(bool): sort the order of the image generator by an integer that is part of the image's
                                         name -> "0.jpg","1.jpg"..

              Returns:
                  np.array: images numpy array with the following shape (number of pictures,rows,columns,channels)
           """
        images_paths = self.__get_images_path(directory, image_type, integer_sorting)
        img_arrays = list()
        for image_path in images_paths:
            image_array = cv2.imread(image_path)
            image_array = cv2.resize(image_array, image_size)
            img_arrays.append(image_array)
        return np.asarray(img_arrays)

    ####################################
    # Private
    ####################################

    @staticmethod
    def __create_image_generator_object(image_generator_configuration: Union[dict, None] = None) -> ImageDataGenerator:
        """Creates a  base image data generator
              Args:
                  image_generator_configuration(dict, None): data structure that contains the initialized members of the
                                                             image data generator
                   Example: generator1 : {"zca_epsilon": 1e-6,"width_shift_range": 0.4}

              Returns:
                  ImageDataGenerator: the created image data generator
           """
        if image_generator_configuration:
            img_gen = ImageDataGenerator(**image_generator_configuration)
        else:
            img_gen = ImageDataGenerator()
        return img_gen

    @staticmethod
    def __create_configured_generators(image_generators_struct: Union[dict, None] = None) -> dict:
        """Creates a dictionary of image data generator
              Args:
                  image_generators_struct(dict, None): data structure that contains the initialized members of the
                                                       image data generator, if None an empty structure will be returned

              Returns:
                  dict:  dictionary of image data generator
           """
        if image_generators_struct is None:
            return dict()
        generators_struct = dict()
        for generator_name, generator_conf in image_generators_struct.items():
            if generator_conf:
                generators_struct[generator_name] = ImageDataGenerator(**generator_conf)
            else:
                generators_struct[generator_name] = ImageDataGenerator()
        return generators_struct

    @staticmethod
    def __get_images_path(directory: str, image_type: str, integer_sorting: bool = False) -> list:
        """Creates a  dictionary of image data generator
              Args:
                  directory(str): the image's folder directory
                  image_type(str): image type -> "jpg","png"....
                  integer_sorting(bool): if true: sort the order of the image generator by an integer that is part of
                                         the image's name -> "0.jpg","1.jpg"..

              Returns:
                  list:  images path list
           """
        image_list_paths = list()
        for image_path in os.listdir(directory):
            if re.search(image_type, image_path):
                image_list_paths.append(os.path.join(directory, image_path))
        if integer_sorting:
            return sorted(image_list_paths, key=lambda img_path: int(os.path.basename(img_path).split(".")[0]))
        else:
            return image_list_paths
