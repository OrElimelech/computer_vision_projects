###########################################################################
# Imports
###########################################################################
import os
import re
import cv2
import numpy as np
from tqdm import tqdm
from typing import Union

###########################################################################
# Global Parameters
###########################################################################

IMAGE_BETA_WEIGHT = IMAGE_ALPHA_WEIGHT = 1
IMAGE_GAMMA_WEIGHT = 0


###########################################################################
# NucleiDataSetArranger Class
###########################################################################

class NucleiDataSetArranger:
    """An object arranges the nuclei data set into convenient data structure
        Args:
            data_set_path(str): path to nuclei data set
    """

    def __init__(self, data_set_path: str = 'U_NET'):
        assert os.path.exists(data_set_path), f"input path does not exist: {data_set_path}"
        self.__data_set_path = data_set_path

        self._train_mask_images = None
        self._train_original_images = None
        self._test_images = None

    ####################################
    # Public
    ####################################

    def retrieve_training_set(self) -> None:
        """Retrieves the training part of the data set - masks and their original images"""
        self._train_mask_images = self.__create_image_struct('train', 'masks', 'png', self.__create_united_mask_image,
                                                             process_name="Mask Images Assembly")
        self._train_original_images = self.__create_image_struct('train', 'images', 'png', cv2.imread,
                                                                 store_in_list=False, process_name="Original "
                                                                                                   "Images Retrieving")

    def retrieve_test_set(self) -> None:
        """Retrieves the testing part of the data set - masks and their original images"""
        self._test_images = self.__create_image_struct('test', 'images', 'png', cv2.imread, store_in_list=False,
                                                       process_name="Test Images Retrieving")

    @property
    def get_training_original_images(self) -> dict:
        """Returns data structure of the training original images

        Returns:
           dict : structure of the training original
        """
        return self._train_original_images

    @property
    def get_training_mask_images(self) -> dict:
        """Returns data structure of the training mask images

        Returns:
           dict : structure of the mask original
        """
        return self._train_mask_images

    @property
    def get_test_images(self) -> dict:
        """Returns data structure of the testing images

        Returns:
           dict : structure of the testing images
        """
        return self._test_images

    ####################################
    # Private
    ####################################

    def __create_image_struct(self, root_directory: str, images_type_dir: str, image_file_type: str,
                              image_reading_function: any,
                              store_in_list: bool = True, process_name: Union[None, str] = None) -> dict:
        """Creates an organized image structure

        Args:
            root_directory(str): root directory relative to the main directory
            images_type_dir(str): images type - can 'images' or 'masks'
            image_file_type(str): images file format (png)
            image_reading_function(any): function that reads the image into a readable structure
            process_name(str/None): name of process
            store_in_list(bool): if true value in structure will be list, else string
        Returns:
            dict: organized image structure
         """
        images_path_struct = self.__get_images_paths_struct(root_directory, images_type_dir, image_file_type,
                                                            store_in_list)
        if process_name:
            print(f"Starting {process_name}...")
        return {num_id: image_reading_function(images_paths) for (num_id, images_paths) in
                tqdm(images_path_struct.items())}

    def __get_images_paths_struct(self, root_directory: str, images_type_dir: str, image_file_type: str,
                                  store_in_list: bool = True) -> dict:
        """Creates organized images path structure

        Args:
            root_directory(str): root directory relative to the main directory
            images_type_dir(str): images type - can 'images' or 'masks'
            image_file_type(str): images file format (png)
            store_in_list(bool): if true value in structure will be list, else string
        Returns:
            dict: organized images path structure
         """
        images_dir_count = 0
        images_file_dict = dict()
        root_directory = os.path.join(self.__data_set_path, root_directory)
        for root, dirs, f in os.walk(root_directory):
            for dir_name in dirs:
                if re.search(images_type_dir, dir_name):
                    curr_root_dir = os.path.join(root, dir_name)
                    images_file_list = [os.path.join(curr_root_dir, image_name) for image_name in
                                        os.listdir(curr_root_dir) if re.search(image_file_type, image_name)]
                    if store_in_list is False:
                        images_file_dict[images_dir_count] = images_file_list[0]
                    elif store_in_list:
                        images_file_dict[images_dir_count] = images_file_list
                    images_dir_count += 1
        return images_file_dict

    @staticmethod
    def __create_united_mask_image(mask_path_list: Union[None, list]) -> Union[np.array, None]:
        """Creates an image array of all the connected masks in the list of paths given in  the input

        Args:
            mask_path_list(list/None): list of paths given in  the input, if None None will be returned
        Returns:
            np.array/None: image numpy array of  the connected masks in the list of paths given in the input, None if
                           input is none
         """
        if not mask_path_list:
            return None
        united_mask_image = np.zeros(cv2.imread(mask_path_list[0]).shape, np.uint8)
        for mask_image_path in mask_path_list:
            united_mask_image = cv2.addWeighted(united_mask_image, IMAGE_ALPHA_WEIGHT,
                                                cv2.imread(mask_image_path), IMAGE_BETA_WEIGHT,
                                                IMAGE_GAMMA_WEIGHT)
        united_mask_image = cv2.cvtColor(united_mask_image, cv2.COLOR_BGR2GRAY)
        united_mask_image = united_mask_image.reshape(united_mask_image.shape[0], united_mask_image.shape[1], 1)
        return united_mask_image
