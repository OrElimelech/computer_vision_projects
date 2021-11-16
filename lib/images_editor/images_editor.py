###########################################################################
# Imports
###########################################################################
import numpy as np
from typing import Union
from copy import deepcopy
from tensorflow import image
from tensorflow import convert_to_tensor


###########################################################################
# ImagesEditor Class
###########################################################################
class ImagesEditor:
    """An object that enables image editing for data preparations purposes
        Args:
            original_image_struct(np.array): numpy array that contains all the images
    """

    def __init__(self, original_image_struct: np.array):
        self.__original_image_struct = original_image_struct
        self.__current_altered_image_struct = original_image_struct
        self.__alterations_struct = self.__build_alterations_dictionary()

    ####################################
    # Public
    ####################################

    def edit_images(self, edit_type: str, *args: any, add_to_image_storage: bool = False,
                    override_originals: bool = False, use_original_struct: bool = False) -> None:
        """Edits the input images according to the required action

        Args:
            edit_type(str): type of action , currently supported -> "resize","color_to_gray","color_to_gray",
                            "flip_left_right","flip_up_down","multi_flip"
            *args(any): arguments for the used functions please check the following to see arguments:
                        tf.image.grayscale_to_rgb,
                        tf.image.rgb_to_grayscale,
                        tf.image.resize
                        image.flip_left_right
                        image.flip_up_down

            add_to_image_storage(bool): if true the created image will be added to the rest of the data storage
            override_originals(bool): if true the altered data will switch the original saved data
            use_original_struct(bool): if true changes will be done on the original data as an input instead of the
                                       previous altered data

        Returns:
            np.array: the confusion matrix structure
         """
        images_list = list()
        if use_original_struct:
            image_struct = deepcopy(self.__original_image_struct)
        else:
            image_struct = deepcopy(self.__current_altered_image_struct)
        try:
            edit_function = self.__alterations_struct[edit_type]
        except KeyError:
            print(f"wrong edit type , the only possible types are: {list(self.__alterations_struct.keys())}")
            return None
        for image_id in image_struct:
            try:
                rescaled_image = edit_function(image_id, *args).numpy()
            except AttributeError:
                image_id = convert_to_tensor(image_id)
                rescaled_image = edit_function(image_id, *args).numpy()
            except ValueError:
                image_id = np.expand_dims(image_id, axis=-1)
                rescaled_image = edit_function(image_id, *args).numpy()
            except TypeError:
                rescaled_image = edit_function(image_id).numpy()
            images_list.append(rescaled_image)
        if add_to_image_storage:
            current_altered_image_list = list(self.__current_altered_image_struct)
            images_list += current_altered_image_list
        self.__current_altered_image_struct = np.asarray(images_list)
        if override_originals:
            self.__original_image_struct = deepcopy(self.__current_altered_image_struct)

    @property
    def get_original_image_struct(self) -> np.array:
        """Returns original input of images, before any alterations

        Returns:
           np.array: original input of images
        """
        return self.__original_image_struct

    @property
    def get_altered_image_struct(self) -> np.array:
        """Returns current altered images

        Returns:
           np.array: array of altered images
        """
        return self.__current_altered_image_struct

    def reset_alterations(self, images_struct: Union[None, np.array] = None) -> None:
        """Resets the altered images structure with the original one or taken a new structure as an input
            Args:
            images_struct(np.array/None): numpy array of images to replace the original and the altered image structure,
                                          if None the altered structure will be replaced by the original structure

        """
        if images_struct is None:
            self.__current_altered_image_struct = deepcopy(self.__original_image_struct)
        else:
            self.__original_image_struct = deepcopy(images_struct)
            self.__current_altered_image_struct = deepcopy(images_struct)

    ####################################
    # Private
    ####################################

    @staticmethod
    def __build_alterations_dictionary() -> dict:
        """Returns a structure that contains the images' editing functions

        Returns:
           dict: structure that contains the images' editing functions
        """
        return {"resize": lambda *args: image.resize(*args),
                "resize_with_crop_or_pad": lambda *args: image.resize_with_crop_or_pad(*args),
                "gray_to_color": lambda *args: image.grayscale_to_rgb(*args),
                "color_to_gray": lambda *args: image.rgb_to_grayscale(*args),
                "flip_left_right": lambda *args: image.flip_left_right(*args),
                "flip_up_down": lambda *args: image.flip_up_down(*args),
                "multi_flip": lambda *args: image.flip_up_down(image.flip_left_right(*args))}
