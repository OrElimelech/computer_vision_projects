###########################################################################
# Imports
###########################################################################
import cv2
import numpy as np
from utilities.drawer import Drawer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import load_model, save_model
from lib.model_builder.fine_tuned_model_builder import FineTunedModelBuilder
from lib.model_analyzer.model_analyzer import ModelAnalyzer
from lib.callback_handler.callback_handler import CallbackHandler
from lib.images_editor.images_editor import ImagesEditor
from lib.configuration_handler.configuration_handler import ConfigurationHandler
from lib.image_generator_handler.image_generator_handler import ImageGeneratorHandler
from lib.segmented_image_contour_handler.segmented_image_contour_handler import SegmentedImagesContourHandler


###########################################################################
# DataReshapeDigitsRecognition Class
###########################################################################

class DataReshapeDigitsRecognition(ImagesEditor):
    """An object that handles the MNIST data set reshaping requirements
        Args:
            image_struct(np.array): numpy images array
            expected_input_shape(tuple): future shape of image - (rows,columns,channels)
    """

    def __init__(self, image_struct: np.array, expected_input_shape: tuple):
        self.__expected_input_shape = expected_input_shape
        super().__init__(image_struct)

    def reshape_digits_data(self) -> None:
        """Changes the images' dimensions and channel depends on the input of the image"""
        self.edit_images("resize", (self.__expected_input_shape[0], self.__expected_input_shape[1]))
        if self.__expected_input_shape[2] == RGB_CHANNELS:
            self.edit_images("gray_to_color", "gray_2_color")


###########################################################################
# Main
###########################################################################


if __name__ == "__main__":
    ###########################################################################
    # General Parameters
    ###########################################################################
    GRAY_SCALE_COLOR = 1
    LINE_WIDTH = 12
    BORDER_PADDING_PERCENTAGE = 0.4
    RGB_CHANNELS = 3
    POSSIBLE_DIGITS = [digit for digit in range(10)]
    CONFUSION_MATRIX_SAMPLES = 1000
    TEXT_PREDICTION_FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_PREDICTION_COLOR = (0, 255, 0)
    PREDICTION_FONT_SCALE = 3
    PREDICTION_LINE_THICKNESS = 5

    ###########################################################################
    # General Configuration Data
    ###########################################################################
    config_handler = ConfigurationHandler()  # loading configuration handler
    data_reshaping_conf = config_handler.get_config_value("data_config")["image_shape"]
    data_augmentation_conf = config_handler.get_config_value("data_config")["data_augmentation"]
    training_conf = config_handler.get_config_value("training_config")
    model_conf = config_handler.get_config_value("model_configuration")
    model_file_name = f"{model_conf['model_name']}_digits_classifier.h5"
    input_size = (data_reshaping_conf["image_rows"],
                  data_reshaping_conf["image_columns"],
                  data_reshaping_conf["channel_numbers"])

    ###########################################################################
    # Building Model
    ###########################################################################

    model_handler = FineTunedModelBuilder(model_conf["model_name"], input_size, model_conf["model_top_config"])
    if training_conf["enable_training"]:
        model_object = model_handler.build_tuned_model()
        model_object.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy, metrics=['accuracy'])
        print(model_object.summary())
    else:
        model_object = load_model(model_file_name)

    ###########################################################################
    # Data Loading
    ###########################################################################
    if training_conf["enable_training"]:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        ###########################################################################
        # Data Reshaping
        ###########################################################################

        # Training Data Set

        image_data_editor = DataReshapeDigitsRecognition(x_train, input_size)
        image_data_editor.reshape_digits_data()
        x_train = image_data_editor.get_altered_image_struct

        # Validation Data Set
        image_data_editor.reset_alterations(x_test)
        image_data_editor.reshape_digits_data()
        x_test = image_data_editor.get_altered_image_struct

        ###########################################################################
        # Data Augmentation and Callbacks
        ###########################################################################
        data_augmentation_conf["preprocessing_function"] = model_handler.get_input_preparation_function()
        data_generator_handler = ImageGeneratorHandler({"training_gen": data_augmentation_conf, "validation_gen": {
            "preprocessing_function": model_handler.get_input_preparation_function()}})
        validation_generator = data_generator_handler.create_generator_from_flow("validation_gen", x_test, y_test, {
            "batch_size": training_conf["batch_size"]})
        training_generator = data_generator_handler.create_generator_from_flow("training_gen", x_train, y_train, {
            "batch_size": training_conf["batch_size"]})
        call_backs_creator = CallbackHandler(training_conf["call_backs_conf"])
        call_backs_creator.build_callback_objects()
        call_backs = call_backs_creator.get_callback_output

        ###########################################################################
        # Model Training
        ###########################################################################
        results = model_object.fit(training_generator, batch_size=training_conf["batch_size"],
                                   epochs=training_conf["epochs"], callbacks=call_backs,
                                   validation_data=validation_generator,
                                   steps_per_epoch=x_train.shape[0] // training_conf["batch_size"],
                                   validation_steps=x_test.shape[0] // training_conf["batch_size"])
        save_model(model_object, model_file_name, overwrite=True)

        ###########################################################################
        # Analyzing Model
        ###########################################################################

        model_analyzer = ModelAnalyzer(results, model_object, "evidence_output_model_analyzer")
        model_analyzer.create_confusion_matrix_graph("Confusion_Matrix_Training.jpg",
                                                     POSSIBLE_DIGITS,
                                                     training_generator, CONFUSION_MATRIX_SAMPLES)
        model_analyzer.create_confusion_matrix_graph("Confusion_Matrix_Validation.jpg", POSSIBLE_DIGITS,
                                                     validation_generator, CONFUSION_MATRIX_SAMPLES)
        model_analyzer.create_data_fitting_graph("validation", "accuracy")
        model_analyzer.create_data_fitting_graph("validation", "loss")
        model_analyzer.create_data_fitting_graph("training", "accuracy")
        model_analyzer.create_data_fitting_graph("training", "loss")

    ###########################################################################
    # Activating the Digits Drawer
    ###########################################################################
    print("################### Draw your Digits with the Mouse ###################")
    print("################### Press a long ESC to Continue ###################")
    print("################### Press a long SPACE to Reset Drawer ###################")
    digit_drawer = Drawer(line_size=LINE_WIDTH)
    assert not digit_drawer.is_image_empty(), "No drawings were found!!!"
    digits_image = digit_drawer.get_current_image

    ###########################################################################
    # Predicting the Drawer's Output
    ###########################################################################
    image_drawer_handler = SegmentedImagesContourHandler("digits_evidence_segmentation")
    image_drawer_handler.get_current_image = digits_image
    image_drawer_handler.calculate_image_contours(sort_mode="LR")
    image_drawer_handler.calculate_contour_bounding_boxes()
    image_drawer_handler.create_evidence_images()
    cropped_images = image_drawer_handler.get_cropped_images
    for image in cropped_images:
        # padding cropped image

        top = bottom = int(image.shape[1] * BORDER_PADDING_PERCENTAGE)
        left = right = int(image.shape[0] * BORDER_PADDING_PERCENTAGE)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
        resized_cropped_image = cv2.resize(image, (input_size[1], input_size[0]))
        resized_cropped_image = cv2.cvtColor(resized_cropped_image, cv2.COLOR_BGR2RGB)
        normalized_image = model_handler.get_input_preparation_function()(resized_cropped_image)

        # creating prediction image based on the result
        prediction_results = model_object.predict(np.expand_dims(normalized_image, axis=0), batch_size=1)
        prediction_image = np.zeros(image.shape)
        text_size = cv2.getTextSize(str(prediction_results.argmax()), TEXT_PREDICTION_FONT, PREDICTION_FONT_SCALE,
                                    PREDICTION_LINE_THICKNESS)[0]
        text_x = (prediction_image.shape[1] - text_size[0]) / 2
        text_y = (prediction_image.shape[0] + text_size[1]) / 2
        cv2.putText(prediction_image, str(prediction_results.argmax()),
                    (int(text_x), int(text_y)), TEXT_PREDICTION_FONT, PREDICTION_FONT_SCALE,
                    TEXT_PREDICTION_COLOR, PREDICTION_LINE_THICKNESS)

        # showing result
        cv2.imshow("Original Digit", image, )
        cv2.imshow("Prediction Digit", prediction_image)
        print("################### Press any Key to Move Forward ###################")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
