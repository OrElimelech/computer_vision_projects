###########################################################################
# imports
###########################################################################
import os
import re
import itertools
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from typing import Union
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt


###########################################################################
# ModelAnalyzer Class
###########################################################################

class ModelAnalyzer:
    """An object that analyzes the model after training
        Args:
            model_training_results(tf.keras.callbacks.History): an object that contains the model results after training
            trained_model(Model): a trained model object
    """

    def __init__(self, model_training_results: tf.keras.callbacks.History,
                 trained_model: Model,
                 output_path: str = '.'):
        self.__trained_model = trained_model
        self.__training_results = model_training_results
        self.__output_path = output_path
        self.__output_path_handler()
        self.__layers_struct = self.__get_convolution_layers_struct()
        self.__confusion_matrix = None

    ####################################
    # Public
    ####################################

    def create_data_fitting_graph(self, data_type: str, graph_type: str) -> None:
        """Creates a graph or loss or accuracy of the model
            Args:
                data_type(str): type of required result data ('validation' or 'training')
                graph_type(str): type of required graph ('loss' or 'accuracy')
         """
        graph_struct = {"validation": {"accuracy": "val_accuracy",
                                       "loss": "val_loss"},
                        "training": {"accuracy": "accuracy",
                                     "loss": "loss"}
                        }
        try:
            required_result_key = graph_struct[data_type][graph_type]
        except Exception:
            raise ValueError(
                f"wrong values!! given data type:  {data_type}, allowed data types : 'training','validation'\n"
                f"given graph type: {graph_type}, allowed graph types: 'loss', 'accuracy'")
        self.__continuous_graph_plot(self.__training_results.epoch,
                                     self.__training_results.history[required_result_key],
                                     "Epoch ID number",
                                     f"{required_result_key} value",
                                     f"{data_type} {required_result_key} Graph")

    def create_confusion_matrix_graph(self, title: str, classes_list: list, image_generator: np.array,
                                      samples_num: Union[int, None] = None,
                                      hot_encoded_flag: bool = False):
        """Creates a graph of the confusion matrix of the trained model
            Args:
                title(str): graph title
                classes_list(list): list of classes name
                image_generator(np.array): image data generator
                samples_num(int/None): number of samples, if none this will be equal to the number of images in the
                                       image generator
                hot_encoded_flag(bool): deals with hot encoded classification
         """
        confusion_matrix_struct = self.__calculate_confusion_matrix(image_generator=image_generator,
                                                                    samples_num=samples_num,
                                                                    hot_encoded_flag=hot_encoded_flag)
        plt.figure(figsize=(15, 15))
        plt.imshow(confusion_matrix_struct, interpolation='nearest', cmap="Blues")
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes_list))
        plt.xticks(tick_marks, classes_list, rotation=45)
        plt.yticks(tick_marks, classes_list)
        plt.tight_layout()
        thresh = confusion_matrix_struct.max() / 2.
        for row, col in itertools.product(range(confusion_matrix_struct.shape[0]),
                                          range(confusion_matrix_struct.shape[1])):
            plt.text(col, row, format(confusion_matrix_struct[row, col], '.2f'),
                     horizontalalignment="center",
                     color="white" if confusion_matrix_struct[row, col] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(self.__output_path, f"{title}.jpg"))
        plt.close()

    def create_feature_maps_plots(self, input_processed_image: Tensor,
                                  blocks_names: Union[list, None] = None):
        """Creates plots of feature maps for a required convolutional layer
            Args:
                input_processed_image(Tensor): processed tensor image
                blocks_names(list): list of layer names
         """
        if blocks_names is None:
            blocks_names = list(self.__layers_struct.keys())
            print("warning: this might be slow and take a lot of memory space!!")
        else:
            self.__validate_layers_names(blocks_names)
        output_path_dir = os.path.join(self.__output_path, "model_feature_maps")
        if not os.path.exists(output_path_dir):
            os.mkdir(output_path_dir)
        blocks_ids = [self.__layers_struct[layer_name] for layer_name in blocks_names]
        outputs = [self.__trained_model.layers[layer_id].output for layer_id in blocks_ids]
        temp_model = Model(inputs=self.__trained_model.inputs, outputs=outputs)
        feature_maps = temp_model.predict(input_processed_image)
        for block_id, f_map in zip(blocks_ids, feature_maps):
            for i in range(1, f_map.shape[3] + 1):
                plt.imshow(f_map[0, :, :, i - 1], cmap='gray')
                plt.savefig(os.path.join(output_path_dir, f"BLOCK_{block_id}_feature_map_number_{i}.jpg"))
                plt.close()

    def create_convolutional_filters_plots(self, filters_samples_number: int, channels_number: int,
                                           convolutional_layers: Union[list, None] = None) -> None:
        """Creates plots of filters for a required convolutional layer
            Args:
                convolutional_layers(list/None): list of convolutional layers, if none all convolutional layers
                                                 will be considered, this is not recommended
                filters_samples_number(int): number of filter samples
                channels_number(int): number of input channels
         """
        if convolutional_layers is None:
            convolutional_layers = list(self.__layers_struct.keys())
            print("warning: this might be slow and take a lot of memory space!!")
        else:
            self.__validate_layers_names(convolutional_layers)

        for layer_struct in self.__trained_model.layers:
            if layer_struct.name in convolutional_layers:
                filters, biases = layer_struct.get_weights()
                filter_min, filter_max = filters.min(), filters.max()
                filters = (filters - filter_min) / (filter_max - filter_min)
                layer_depth = filters.shape[3]
                assert filters_samples_number <= layer_depth, f"number of required samples is bigger than " \
                                                              f"the expected layer depth->" \
                                                              f" layer depth: {layer_depth} , " \
                                                              f"required samples: {filters_samples_number}"
                output_path_dir = os.path.join(self.__output_path, "model_filters")
                if not os.path.exists(output_path_dir):
                    os.mkdir(output_path_dir)
                for filter_sample_id in range(filters_samples_number):
                    filter_data = filters[:, :, :, filter_sample_id]
                    for channel_num in range(channels_number):
                        title_name = f"{layer_struct.name} filter - sample number {filter_sample_id} - " \
                                     f"channel number {channel_num}"
                        plt.title(title_name)
                        plt.imshow(filter_data[:, :, channel_num], cmap='gray')
                        plt.savefig(os.path.join(output_path_dir, f"{title_name}.jpg"))

    @property
    def get_layer_struct(self) -> dict:
        """Returns the convolutional layer structure of the model

        Returns:
           dict: convolutional layer structure
        """
        return self.__layers_struct

    ####################################
    # Private
    ####################################

    def __calculate_confusion_matrix(self, image_generator: np.array, samples_num: Union[int, None] = None,
                                     hot_encoded_flag: bool = False) -> np.array:
        """Calculates the confusion matrix structure.

        Args:
            image_generator(np.array): image date generator
            samples_num(int/None): number of samples, if none this will be equal to the number of images in the
                                       image generator
            hot_encoded_flag(bool): deals with hot encoded classification

        Returns:
            np.array: the confusion matrix structure
         """
        predictions, targets = list(), list()
        sample_threshold = image_generator.n if not samples_num else samples_num
        for image, image_label in image_generator:
            image_prediction = self.__trained_model.predict(image)
            if hot_encoded_flag:
                found_label = np.argmax(image_label, axis=1)
            else:
                found_label = image_label
            predictions = np.concatenate((predictions, np.argmax(image_prediction, axis=1)))
            targets = np.concatenate((targets, found_label))
            image_count = len(targets)
            print(f"current image count for confusion matrix calculation: {image_count}")
            if image_count >= sample_threshold:
                break
        return confusion_matrix(targets, predictions)

    def __continuous_graph_plot(self, x_axis: Union[np.array, list], y_axis: Union[np.array, list], x_label: str,
                                y_label: str, graph_title: str) -> None:
        """Creates a continuous graph plot

        Args:
            x_axis(np.array/list): x axis values
            y_axis(np.array/list): y axis values
            x_label(str): x axis title
            y_label(str): y axis title

         """
        plt.plot(x_axis, y_axis)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(x_axis)
        plt.title(graph_title)
        plt.grid()
        plt.savefig(os.path.join(self.__output_path, f"{graph_title}.jpg"))
        plt.close()

    def __output_path_handler(self) -> None:
        """Handles the input path for the evidence folder, if does not exist it will be crated"""
        if not os.path.exists(self.__output_path):
            os.mkdir(self.__output_path)

    def __get_convolution_layers_struct(self) -> dict:
        """Builds the convolutional layer structure of the model

        Returns:
           dict: convolutional layer structure
        """
        return {layer.name: layer_index for layer_index, layer in enumerate(self.__trained_model.layers) if
                re.search("conv", layer.name)}

    def __validate_layers_names(self, convolutional_layers: list) -> None:
        """Validates the existence of the given layer names

        Args:
            convolutional_layers(list): convolutional layers name list

         """
        for layer_name in convolutional_layers:
            assert layer_name in list(
                self.__layers_struct.keys()), f"invalid layer name: {layer_name}, all layer names possibilities: " \
                                              f"{list(self.__layers_struct.keys())}"
