###########################################################################
# Imports
###########################################################################
import os
import cv2
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.losses import binary_crossentropy, mse
from lib.model_analyzer.model_analyzer import ModelAnalyzer
from lib.callback_handler.callback_handler import CallbackHandler
from lib.configuration_handler.configuration_handler import ConfigurationHandler
from lib.model_builder.fine_tuned_model_builder import FineTunedModelBuilder
from utilities.image_car_plate_data_generator import ImageCarPlateDataGenerator

###########################################################################
# Main
###########################################################################


if __name__ == "__main__":
    ###########################################################################
    # General Parameters
    ###########################################################################
    NORMALIZATION_FACTOR = 255
    RGB_CHANNELS = 3
    RECT_COLOR = (0, 255, 0)
    RECT_WIDTH = 4
    DATA_SET_URL = "https://www.kaggle.com/andrewmvd/car-plate-detection"
    CAR_PLATE_DETECTOR_FILE = "car_plate_localization.h5"
    TEXT_PREDICTION_FONT = cv2.FONT_HERSHEY_SIMPLEX
    NO_OBJECT_COLOR = (0, 0, 255)
    PREDICTION_FONT_SCALE = 0.5
    PREDICTION_LINE_THICKNESS = 2
    NO_CAR_PLATE_TEXT = "Car Plate Was not Found!!"

    ###########################################################################
    # General Configuration Data
    ###########################################################################
    config_handler = ConfigurationHandler()  # loading configuration handler
    data_set_conf = config_handler.get_config_value("data_set_conf")
    training_conf = config_handler.get_config_value("training_conf")
    model_conf = config_handler.get_config_value("model_conf")
    call_back_conf = config_handler.get_config_value("call_back_conf")


    ###########################################################################
    # Car Plate Localization Loss Function
    ###########################################################################

    def car_plate_localization_loss(y_true, y_pred,
                                    bbox_loss=training_conf["bounding_box_loss_weight"],
                                    obj_exist_loss=training_conf[
                                        "object_existence_loss_weight"]):

        loss_bounding_box = mse(y_true[:, 1:], y_pred[:, 1:])
        loss_obj_found = binary_crossentropy(y_true[:, 0], y_pred[:, 0])
        return bbox_loss * loss_bounding_box * y_true[:, 0] + obj_exist_loss * loss_obj_found


    ###########################################################################
    # Data Loading
    ###########################################################################
    print(f"Data set can be Downloaded here: {DATA_SET_URL}")
    image_size = (data_set_conf["data_dimension"], data_set_conf["data_dimension"])
    validation_data_generator = ImageCarPlateDataGenerator(data_set_conf["images_path"],
                                                           data_set_conf["annotations_path"],
                                                           image_size,
                                                           lambda image_input: image_input / NORMALIZATION_FACTOR,
                                                           only_background=True,
                                                           crop_percentage_index=abs(1 -
                                                                                     training_conf[
                                                                                         "validation_ratio"]),
                                                           crop_from_start_flag=False)
    if training_conf["enable_training"]:
        training_data_generator = ImageCarPlateDataGenerator(data_set_conf["images_path"],
                                                             data_set_conf["annotations_path"],
                                                             image_size,
                                                             lambda image_input: image_input / NORMALIZATION_FACTOR,
                                                             add_vertical_flip=True, only_background=True,
                                                             crop_percentage_index=abs(
                                                                 1 - training_conf["validation_ratio"]))

        ###########################################################################
        # Building Model
        ###########################################################################
        call_backs_creator = CallbackHandler(call_back_conf)
        call_backs_creator.build_callback_objects()
        call_backs = call_backs_creator.get_callback_output
        image_shape = (image_size[0], image_size[1], RGB_CHANNELS)
        model_handler = FineTunedModelBuilder(model_conf["model_type"], image_shape, model_conf["model_top_config"])
        model_object = model_handler.build_tuned_model()
        model_object.compile(loss=car_plate_localization_loss,
                             optimizer=Adam(learning_rate=training_conf["optimizer_learning_rate"]))
        print(model_object.summary())

        ###########################################################################
        # Model Training
        ###########################################################################
        results = model_object.fit(training_data_generator, epochs=training_conf["epoch_num"],
                                   validation_data=validation_data_generator,
                                   steps_per_epoch=training_data_generator.get_max_generated_images // training_conf[
                                       "batch_size"],
                                   validation_steps=validation_data_generator.get_max_generated_images,
                                   callbacks=call_backs)
        save_model(model_object, CAR_PLATE_DETECTOR_FILE, overwrite=True)

        ###########################################################################
        # Analyzing Model
        ###########################################################################

        model_analyzer = ModelAnalyzer(results, model_object, "evidence_output_model_analyzer")
        model_analyzer.create_data_fitting_graph("validation", "loss")
        model_analyzer.create_data_fitting_graph("training", "loss")
    else:
        model_object = load_model(CAR_PLATE_DETECTOR_FILE, compile=False)
        model_object.compile(loss=car_plate_localization_loss,
                             optimizer=Adam(learning_rate=training_conf["optimizer_learning_rate"]))

    ###########################################################################
    # Predictions
    ###########################################################################
    for image_name in os.listdir('test_images'):
        image = cv2.imread(os.path.join('test_images', image_name))
        image = cv2.resize(image, image_size)
        prediction = model_object.predict(np.expand_dims(image / NORMALIZATION_FACTOR, axis=0))[0]
        if round(prediction[0]) == 1:
            x_min = round(prediction[1] * image.shape[1])
            x_max = round(prediction[2] * image.shape[1])
            y_min = round(prediction[3] * image.shape[0])
            y_max = round(prediction[4] * image.shape[0])
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), RECT_COLOR, RECT_WIDTH)
        else:
            text_size = cv2.getTextSize(NO_CAR_PLATE_TEXT, TEXT_PREDICTION_FONT, PREDICTION_FONT_SCALE,
                                        PREDICTION_LINE_THICKNESS)[0]
            text_x = (image.shape[1] - text_size[0]) / 2
            text_y = (image.shape[0] + text_size[1]) / 2
            cv2.putText(image, NO_CAR_PLATE_TEXT,
                        (int(text_x), int(text_y)), TEXT_PREDICTION_FONT, PREDICTION_FONT_SCALE,
                        NO_OBJECT_COLOR, PREDICTION_LINE_THICKNESS)
        cv2.imshow("Prediction", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()