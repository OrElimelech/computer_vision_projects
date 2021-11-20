###########################################################################
# Imports
###########################################################################
import cv2
import numpy as np
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.optimizers import Adam
from utilities.nuclei_dataset_arranger import NucleiDataSetArranger
from utilities.u_net_model_config_structure_creator import UnetModelConfigStructureCreator
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.losses import binary_crossentropy
from lib.images_editor.images_editor import ImagesEditor
from lib.model_builder.model_builder import ModelBuilder
from lib.callback_handler.callback_handler import CallbackHandler
from lib.configuration_handler.configuration_handler import ConfigurationHandler


###########################################################################
# NucleiDataReshape Class
###########################################################################
class NucleiDataReshape(ImagesEditor):
    """An object that handles nuclei segmentation data set required reshaping
        Args:
            image_struct(dict): images data structure
            image_expected_rows(int): future number of image rows
            image_expected_columns(int): future number of image columns
    """

    def __init__(self, image_struct: dict, image_expected_rows: int, image_expected_columns: int):
        super().__init__(list(image_struct.values()))
        self.__image_expected_rows = image_expected_rows
        self.__image_expected_columns = image_expected_columns

    ####################################
    # Public
    ####################################

    def nuclei_data_reshape(self) -> None:
        """Changes the data shape and alter it for better model training"""
        self.edit_images("resize_with_crop_or_pad", self.__image_expected_rows, self.__image_expected_columns,
                         override_originals=True)
        self.edit_images("flip_left_right", add_to_image_storage=True, use_original_struct=True)
        self.edit_images("flip_up_down", add_to_image_storage=True, use_original_struct=True)
        self.edit_images("multi_flip", add_to_image_storage=True, use_original_struct=True)


###########################################################################
# Main
###########################################################################


if __name__ == "__main__":
    ###########################################################################
    # General Parameters
    ###########################################################################
    MODEL_FILE_NAME = "nuclei_segmentation_detector.h5"
    IMAGE_ROWS = 256
    IMAGE_COLS = 256
    GRAY_SCALE_CHANNEL = 1
    RGB_CHANNEL = 3
    DATA_SET_URL = "https://www.kaggle.com/espsiyam/nuclei-image-segmentation"
    Q_ASCII_VALUE = 113
    PIXEL_THRESHOLD = 0.5

    ###########################################################################
    # General Configuration Data
    ###########################################################################
    config_handler = ConfigurationHandler()
    data_set_path = config_handler.get_config_value("data_set_location")
    training_conf = config_handler.get_config_value("training_configuration")

    ###########################################################################
    # Data Arranging
    ###########################################################################
    print(f"Data set can be Downloaded here: {DATA_SET_URL}")
    data_arranger_obj = NucleiDataSetArranger(data_set_path)
    data_arranger_obj.retrieve_test_set()
    test_data_struct = data_arranger_obj.get_test_images

    ###########################################################################
    # Building Model
    ###########################################################################
    if training_conf["enable_training"]:
        model_build_config = config_handler.get_config_value("model_structure")
        model_conf_builder = UnetModelConfigStructureCreator(model_build_config["encoder_params"],
                                                             model_build_config["bottle_neck_params"],
                                                             model_build_config["decoder_params"])
        model_conf_struct = model_conf_builder.build_model_configuration()
        input_layer_struct = {"initializing_parameters": {"shape": (IMAGE_ROWS, IMAGE_COLS, RGB_CHANNEL)},
                              "layer_type": "Input"}

        model_builder = ModelBuilder(model_conf_struct, input_layer_struct)
        model_obj = model_builder.build_model()
        model_obj.compile(optimizer=Adam(), loss=binary_crossentropy, metrics=[MeanIoU(num_classes=2)])

        ###########################################################################
        # Data Reshaping And Augmentation
        ###########################################################################

        # retrieving training data
        data_arranger_obj.retrieve_training_set()
        mask_images_struct = data_arranger_obj.get_training_mask_images
        original_images_struct = data_arranger_obj.get_training_original_images

        # handling mask images
        mask_image_editor = NucleiDataReshape(mask_images_struct,
                                              IMAGE_ROWS,
                                              IMAGE_COLS)
        mask_image_editor.nuclei_data_reshape()
        mask_images = mask_image_editor.get_altered_image_struct / 255.

        # handling original images
        original_image_editor = NucleiDataReshape(original_images_struct,
                                                  IMAGE_ROWS,
                                                  IMAGE_COLS)
        original_image_editor.nuclei_data_reshape()
        original_images = original_image_editor.get_altered_image_struct / 255.

        ###########################################################################
        # Training Model
        ###########################################################################
        print(model_obj.summary())
        call_backs_creator = CallbackHandler(training_conf["call_backs_conf"])
        call_backs_creator.build_callback_objects()
        call_backs = call_backs_creator.get_callback_output
        training_params = training_conf["training_parameters"]
        model_obj.fit(original_images.astype(dtype=np.uint8), mask_images.astype(dtype=bool),
                      training_params["batch_size"], training_params["epoch_num"],
                      validation_split=training_params["validation_split_ratio"], callbacks=call_backs)
        save_model(model_obj, MODEL_FILE_NAME, overwrite=True)

    else:
        ###########################################################################
        # Loading Model
        ###########################################################################
        model_obj = load_model(MODEL_FILE_NAME)

    ###########################################################################
    # Predictions
    ###########################################################################
    print("################### Press 'q' to End Prediction Process or Any Other Key to Continue ###################")
    for image_id, image in test_data_struct.items():
        image = cv2.resize(image, (IMAGE_COLS, IMAGE_ROWS))
        mask_prediction = model_obj.predict(np.expand_dims(image, axis=0), batch_size=1)
        mask_prediction = (mask_prediction[0]).astype(np.uint8) * 255
        cv2.imshow(f"Original Image Number {image_id}", image)
        cv2.imshow(f"Mask Prediction {image_id}", mask_prediction)
        button_val = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if button_val == Q_ASCII_VALUE:
            break
    print("################### Prediction Process is Done ###################")
