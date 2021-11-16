###########################################################################
# imports
###########################################################################
from typing import Union, Callable
from tensorflow import expand_dims
from tensorflow import Tensor
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, BatchNormalization, \
    Dense, Dropout, Flatten, Activation, MaxPooling2D, Input, ZeroPadding2D, add


###########################################################################
# ModelBuilder Class
###########################################################################

class ModelBuilder:
    """Builds a dynamic model according to given configuration
        Args:
            model_instructions(dict): a configuration structure of the inner model body
            input_layer_instructions: a configuration structure of the input model layer

        Example:
            model_instructions = {"conv2d_1": {"initializing_parameters": {"filters": 32,
                                                                            "kernel_size": (3, 3),
                                                                            "strides": 2,
                                                                            "activation": 'relu'},
                                  "layer_type": "Conv2D"},
                                  "dense_1": {"initializing_parameters": {"units": 10,
                                                                          "activation":
                                                                          'softmax'},
                                              "input_layer": "conv2d_1",
                                              "layer_type": "Dense"}}
            input_layer_instructions = {"initializing_parameters": {"shape": (28, 28, 1)}, "layer_type": "Input"}

    """

    def __init__(self, model_instructions: dict, input_layer_instructions: dict):
        self.__layer_struct = self.__build_layer_struct()
        self.__model_instructions = model_instructions
        self.__input_layer_instructions = input_layer_instructions

    ####################################
    # Protected
    ####################################

    def _get_initialized_layer(self, layer_name: str, initializing_parameters_struct: dict) -> Union[Conv2D,
                                                                                                     Conv2DTranspose,
                                                                                                     Concatenate,
                                                                                                     BatchNormalization,
                                                                                                     Dense, Dropout,
                                                                                                     Flatten,
                                                                                                     Activation,
                                                                                                     MaxPooling2D,
                                                                                                     Input,
                                                                                                     ZeroPadding2D,
                                                                                                     add]:
        """Creates a layer according to configuration
            Args:
                layer_name(str): name of the layer type - check the __build_layer_struct() method
                initializing_parameters_struct(dict): initialization parameters for the layer

            Returns:
                layer: an initialized layer
         """
        if initializing_parameters_struct:
            return self.__layer_struct[layer_name](**initializing_parameters_struct)
        else:
            return self.__layer_struct[layer_name]()

    def _build_model_body(self, initial_input_layer: any) -> dict:
        """Creates a layer according to configuration
            Args:
                initial_input_layer(any): initialized input layer

            Returns:
                dict: contains all the initialized layers of the inner layers of the model
         """
        initial_input_layer_addition = False
        layer_history = dict()
        layer_inputs = None
        for layer_name, layer_data in self.__model_instructions.items():
            if not initial_input_layer_addition:
                layer_history[layer_name] = self._get_initialized_layer(layer_data["layer_type"],
                                                                        layer_data["initializing_parameters"])(
                    initial_input_layer)
                initial_input_layer_addition = True
            else:
                if isinstance(layer_data["input_layer"], str):
                    layer_inputs = layer_history[layer_data["input_layer"]]
                elif isinstance(layer_data["input_layer"], list):
                    layer_inputs = [layer_history[layer_name] for layer_name in layer_data["input_layer"]]
                layer_history[layer_name] = self._get_initialized_layer(layer_data["layer_type"],
                                                                        layer_data["initializing_parameters"])(
                    layer_inputs)
        return layer_history

    ####################################
    # Private
    ####################################

    @staticmethod
    def __build_layer_struct() -> dict:
        """Creates the structure that contains all possible layers that can be used

        Returns:
            dict: data structure with all possible layers

         """
        return {"Conv2D": Conv2D,
                "Conv2DTranspose": Conv2DTranspose,
                "Concatenate": Concatenate,
                "BatchNormalization": BatchNormalization,
                "Dense": Dense,
                "Dropout": Dropout,
                "Flatten": Flatten,
                "Activation": Activation,
                "MaxPooling2D": MaxPooling2D,
                "Input": Input,
                "ZeroPadding2D": ZeroPadding2D,
                "add": add}

    ####################################
    # Public
    ####################################

    def build_model(self) -> Model:
        """Creates a complete model according to configuration.

        Returns:
            Model: initialized model object
         """
        input_layer = self._get_initialized_layer(self.__input_layer_instructions["layer_type"],
                                                  self.__input_layer_instructions["initializing_parameters"])
        layer_history = self._build_model_body(input_layer)
        return Model(input_layer, list(layer_history.values())[-1])

    @staticmethod
    def prepare_image(image_path: str, target_size: tuple, grayscale_mode: bool = False,
                      preparation_function: Union[Callable, None] = None) -> Tensor:
        """Prepare an image to be an input  as a tensor
            Args:
                image_path(str): path to the given image file
                target_size(tuple): the required size of the image
                grayscale_mode(bool): true, will be turned to grayscale image, false - will remain the same
                preparation_function(None/Callable): a normalization function, depends on the model, if None image
                                                     tensor will remain the same

            Returns:
                Tensor: image tensor
         """
        image = load_img(image_path, target_size=target_size, grayscale=grayscale_mode)
        image = img_to_array(image)
        image = expand_dims(image, axis=0)
        if preparation_function:
            return preparation_function(image)
        else:
            return image

    @property
    def get_layer_struct(self) -> dict:
        """Returns the structure that contains all possible layers that can be used

        Returns:
            dict: data structure with all possible layers
         """
        return self.__layer_struct
