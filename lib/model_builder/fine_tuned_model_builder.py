###########################################################################
# imports
###########################################################################
from typing import Union, Tuple
from tensorflow import Tensor
from tensorflow.keras.models import Model
from lib.model_builder.model_builder import ModelBuilder
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_data_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_data_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_data_preprocess


###########################################################################
# FineTunedModelBuilder Class
###########################################################################
class FineTunedModelBuilder(ModelBuilder):
    """Builds a fine - tuned model according to given configuration
        Args:
            model_type(str): model name from current options : 'VGG19', 'VGG16', 'ResNet50'
            input_shape(list/Tuple): input shape
            top_model_instructions(dict): data structure that contains the instructions for building the model's top
            weights(str): path to weights file

        Example:
            model_type = "VGG16
            input_shape = [64,64,1] (or =(64,64,1))
            weights = <path to weights file>
            top_model_instructions = {"Flatten_Layer": {"initializing_parameters": {},
                                                        "layer_type": "Flatten"},
                                      "dense_1": {"initializing_parameters": {"units": 1000,
                                                                              "activation": 'relu'},
                                                  "input_layer": "Flatten_Layer",
                                                  "layer_type": "Dense"},
                                      "dense_2": {"initializing_parameters": {"units": 10,
                                                                          "activation":
                                                                          'softmax'},
                                                  "input_layer": "dense_1",
                                                  "layer_type": "Dense"}}

    """

    def __init__(self, model_type: str, input_shape: Union[list, Tuple], top_model_instructions: dict,
                 weights: str = 'imagenet'):

        self.__model_type = model_type
        self.__input_shape = input_shape
        self.__model_weights = weights
        self.__top_model_instructions = top_model_instructions
        self.__model_struct = self.__get_model_struct()
        assert self.__is_model_type_valid()
        super().__init__(self.__top_model_instructions, dict())

    ####################################
    # Public
    ####################################

    def prepare_image_fine_tuned(self, image_path: str, target_size: tuple, grayscale_mode: bool = False) -> Tensor:
        """Prepare an image to be an input as a normalized tensor
            Args:
                image_path(str): path to the given image file
                target_size(tuple): the required size of the image
                grayscale_mode(bool): true, will be turned to grayscale image, false - will remain the same

            Returns:
                Tensor: normalized image tensor
         """
        img = self.prepare_image(image_path=image_path, target_size=target_size, grayscale_mode=grayscale_mode,
                                 preparation_function=self.__model_struct[self.__model_type]["preparation_function"])
        return img

    def build_tuned_model(self) -> Model:
        """Creates a fine - tuned model object

        Returns:
           Model: frozen model
        """
        initial_model = self.__get_model()
        layer_history = self._build_model_body(initial_model.output)
        return Model(initial_model.input, list(layer_history.values())[-1])

    def get_input_preparation_function(self) -> callable:
        """Creates an input preparation function relevant for the required model

        Returns:
           callable: not initialized function
        """
        return self.__model_struct[self.__model_type]["preparation_function"]

    def build_model(self):
        """blocked function from the inherited class"""

        raise AttributeError("'FineTunedModel' object has no attribute 'build_model'")

    ####################################
    # Private
    ####################################

    def __get_model(self) -> Union[VGG19, VGG16, ResNet50]:
        """Creates a frozen model from the enabled options of : [VGG19, VGG16, ResNet50]

        Returns:
            Union[VGG19, VGG16, ResNet50, Xception, ResNet101V2]: frozen model
         """
        pre_trained_model = self.__model_struct[self.__model_type]["model"](
            include_top=False, input_shape=self.__input_shape, weights=self.__model_weights)
        for layer in pre_trained_model.layers:
            layer.trainable = False
        return pre_trained_model

    def __is_model_type_valid(self) -> bool:
        """validates if the model name is valid and can be created

        Returns:
            bool: true - model name is valid, else - false
         """
        if self.__model_type not in self.__model_struct:
            print(
                f"{self.__model_type} model is not defined , only the following are allowed: "
                f"{list(self.__model_struct.keys())}")
            return False
        else:
            return True

    @staticmethod
    def __get_model_struct() -> dict:
        """Creates the structure that contains all possible models that can be used with a relevant input function

        Returns:
            dict: data structure with all possible models and their input function

         """
        return {"VGG19": {"model": VGG19,
                          "preparation_function": vgg19_data_preprocess},
                "VGG16": {"model": VGG16,
                          "preparation_function": vgg16_data_preprocess},
                "ResNet50": {"model": ResNet50,
                             "preparation_function": resnet50_data_preprocess}}
