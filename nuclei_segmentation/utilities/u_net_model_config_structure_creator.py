###########################################################################
# Imports
###########################################################################
from typing import Union


###########################################################################
# UnetModelConfigStructureCreator Class
###########################################################################


class UnetModelConfigStructureCreator:
    def __init__(self, encoder_struct: dict, bottleneck_struct: dict, decoder_struct: dict):
        """An object that creates  complete configuration structure of a U-net model
            Args:
                encoder_struct(dict): encoder configuration structure
                bottleneck_struct(dict): bottleneck configuration structure
                decoder_struct(dict): decoder configuration structure
        """
        self.__encoder_struct = encoder_struct
        self.__bottleneck_struct = bottleneck_struct
        self.__decoder_struct = decoder_struct
        self.__layer_config_operator = self.__layer_configuration_struct()

    ####################################
    # Public
    ####################################

    def build_model_configuration(self) -> dict:
        """Creates the structure that contains the complete u-net model configuration

        Returns:
            dict: data structure that contains the complete u-net model configuration

         """
        encoder_block_conf = self.__build_encoder()
        bottleneck_block_conf = self.__build_bottleneck("bottle_neck",
                                                        self.__bottleneck_struct["changing_conv_params"][
                                                            "dropout_rate"],
                                                        self.__bottleneck_struct["general_conv_params"],
                                                        self.__bottleneck_struct["changing_conv_params"]["filter_size"],
                                                        list(encoder_block_conf.keys())[-1])
        decoder_block_conf = self.__build_decoder()
        output = self.__layer_config_operator["convolution_layer"](1, {"kernel_size": (1, 1), "activation": 'sigmoid'})
        output["input_layer"] = list(decoder_block_conf.keys())[-1]
        return {**encoder_block_conf, **bottleneck_block_conf, **decoder_block_conf, **{"output_layer": output}}

    ####################################
    # Private
    ####################################

    def __build_decoder(self) -> dict:
        """Creates the structure that contains the decoder configuration

        Returns:
            dict: data structure that contains the decoder configuration

         """
        decoder_config = dict()
        for loop_id, (block_name, block_data) in enumerate(self.__decoder_struct["changing_conv_params"].items()):
            if loop_id != 0:
                initial_input_layer_name = list(decoder_config.keys())[-1]
            else:
                initial_input_layer_name = "bottle_neck_convolution_2"
            block_conf = self.__create_decoder_block(block_name, block_data['dropout_rate'],
                                                     self.__decoder_struct["general_conv_params"],
                                                     block_data['filter_size'],
                                                     self.__decoder_struct['general_transpose_conv_params'],
                                                     f"{block_data['concatenate_layer']}_convolution_2",
                                                     initial_input_layer_name)
            decoder_config = {**decoder_config, **block_conf}
        return decoder_config

    def __build_encoder(self) -> dict:
        """Creates the structure that contains the encoder configuration

        Returns:
            dict: data structure that contains the encoder configuration

         """
        encoder_config = dict()
        for loop_id, (block_name, block_data) in enumerate(self.__encoder_struct["changing_conv_params"].items()):
            if loop_id != 0:
                initial_input_layer_name = list(encoder_config.keys())[-1]
            else:
                initial_input_layer_name = None
            block_conf = self.__create_encoder_block(block_name, block_data['dropout_rate'],
                                                     self.__encoder_struct["general_conv_params"],
                                                     block_data['filter_size'],
                                                     self.__encoder_struct["max_pool_size"],
                                                     initial_input_layer_name)
            encoder_config = {**encoder_config, **block_conf}
        return encoder_config

    def __build_bottleneck(self, block_name: str, drop_out_rate: float, convolution_struct: dict, filters_size: int,
                           initial_input_layer_name: str) -> dict:
        """Creates the structure that contains the bottleneck configuration

        Args:
            block_name(str): name of block
            drop_out_rate(float): dropout probability for the dropout layer of the block
            convolution_struct(dict): data structure that contains the convolution layer configuration of the block
            filters_size(int): filer size of the convolution layers
            initial_input_layer_name(str): name of input layer for the bottleneck block

        Returns:
            dict: data structure that contains the bottleneck configuration

         """
        conv1_struct = self.__layer_config_operator['convolution_layer'](filters_size, convolution_struct)
        conv1_struct['input_layer'] = initial_input_layer_name
        drop_out_struct = self.__layer_config_operator['drop_out_layer'](drop_out_rate)
        conv2_struct = self.__layer_config_operator['convolution_layer'](filters_size, convolution_struct)
        return {f"{block_name}_convolution_1": conv1_struct,
                f"{block_name}_dropout": {**drop_out_struct, **{"input_layer": f"{block_name}_convolution_1"}},
                f"{block_name}_convolution_2": {**conv2_struct, **{"input_layer": f"{block_name}_dropout"}}}

    def __create_encoder_block(self, block_name: str, drop_out_rate: float, convolution_struct: dict, filters_size: int,
                               max_pool_size: list,
                               initial_input_layer_name: Union[str, None] = None) -> dict:
        """Creates the structure that contains a encoder block configuration

        Args:
            block_name(str): name of block
            drop_out_rate(float): dropout probability for the dropout layer of the block
            convolution_struct(dict): data structure that contains the convolution layer configuration of the block
            filters_size(int): filer size of the convolution layers
            initial_input_layer_name(str): name of input layer for the bottleneck block, if None no input will be
                                           considered
            max_pool_size(list): size of the max pooling layer

        Returns:
            dict: data structure that contains an encoder configuration
         """
        conv1_struct = self.__layer_config_operator['convolution_layer'](filters_size, convolution_struct)
        if initial_input_layer_name:
            conv1_struct['input_layer'] = initial_input_layer_name
        drop_out_struct = self.__layer_config_operator['drop_out_layer'](drop_out_rate)
        conv2_struct = self.__layer_config_operator['convolution_layer'](filters_size, convolution_struct)
        max_pool_struct = self.__layer_config_operator['max_pool_layer'](max_pool_size)
        return {f"{block_name}_convolution_1": conv1_struct,
                f"{block_name}_dropout": {**drop_out_struct, **{"input_layer": f"{block_name}_convolution_1"}},
                f"{block_name}_convolution_2": {**conv2_struct, **{"input_layer": f"{block_name}_dropout"}},
                f"{block_name}_max_pool": {**max_pool_struct, **{"input_layer": f"{block_name}_convolution_2"}}}

    def __create_decoder_block(self, block_name: str, drop_out_rate: float, convolution_struct: dict, filters_size: int,
                               deconvolution_struct: dict,
                               concatenate_layer: str, initial_input_layer_name: Union[str, None] = None) -> dict:
        """Creates the structure that contains a decoder block configuration

        Args:
            block_name(str): name of block
            drop_out_rate(float): dropout probability for the dropout layer of the block
            convolution_struct(dict): data structure that contains the convolution layer configuration of the block
            filters_size(int): filer size of the convolution layers
            initial_input_layer_name(str): name of input layer for the bottleneck block, if None no input will be
                                           considered
            deconvolution_struct(dict): data structure that contains the transpose convolution layer configuration of
                                        the block
            concatenate_layer(str): name of layer to be concatenated into the decoder block

        Returns:
            dict: data structure that contains a decoder block configuration
         """
        transpose_conv_struct = self.__layer_config_operator['transpose_convolution_layer'](filters_size,
                                                                                            deconvolution_struct)
        if initial_input_layer_name:
            transpose_conv_struct['input_layer'] = initial_input_layer_name
        conv1_struct = self.__layer_config_operator['convolution_layer'](filters_size, convolution_struct)
        drop_out_struct = self.__layer_config_operator['drop_out_layer'](drop_out_rate)
        conv2_struct = self.__layer_config_operator['convolution_layer'](filters_size, convolution_struct)
        return {f"{block_name}_transpose_convolution": transpose_conv_struct,
                f"{block_name}_concatenate": self.__layer_config_operator['concatenate_layer'](
                    [concatenate_layer, f"{block_name}_transpose_convolution"]),
                f"{block_name}_convolution_1": {**conv1_struct, **{"input_layer": f"{block_name}_concatenate"}},
                f"{block_name}_drop_out": {**drop_out_struct, **{"input_layer": f"{block_name}_convolution_1"}},
                f"{block_name}_convolution_2": {**conv2_struct, **{"input_layer": f"{block_name}_drop_out"}}
                }

    @staticmethod
    def __layer_configuration_struct() -> dict:
        """Creates the structure that contains all possible layers that can be used for the U-net model

        Returns:
            dict: data structure with all possible layers

         """
        return {
            "convolution_layer": lambda filters_num, struct: {"layer_type": "Conv2D",
                                                              "initializing_parameters": {**{"filters": filters_num},
                                                                                          **struct}},
            "drop_out_layer": lambda drop_rate: {"layer_type": "Dropout",
                                                 "initializing_parameters": {"rate": drop_rate}},
            "transpose_convolution_layer": lambda filters_num, struct: {"layer_type": "Conv2DTranspose",
                                                                        "initializing_parameters": {
                                                                            **{"filters": filters_num},
                                                                            **struct}},
            "max_pool_layer": lambda max_pool_size: {"layer_type": "MaxPooling2D",
                                                     "initializing_parameters": {"pool_size": max_pool_size}},
            "concatenate_layer": lambda layers_input_names: {"layer_type": "Concatenate", "initializing_parameters": {},
                                                             "input_layer": layers_input_names}

        }
