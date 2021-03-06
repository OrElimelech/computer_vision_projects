# Car Plate Localization


## Description
------
This project utilizes Tensorflow's pre-trained classifiers (I used VGG16) to detect if a car plate is in an image or not.
It also predicts the bounding box of the car plate. This was done in order to explore the capabilities of a pre trained network for object localization.
The outcome can be seen in the results folder.


used data set: "https://www.kaggle.com/andrewmvd/car-plate-detection"
 

## Model Structure
------



| Layer (type)               | Output Shape          | Param   |
|----------------------------|-----------------------|---------|
| input_1 (InputLayer)       | [(None, 400, 400, 3)] | 0       |
| block1_conv1 (Conv2D)      | (None, 400, 400, 64)  | 1792    |
| block1_conv2 (Conv2D)      | (None, 400, 400, 64)  | 36928   |
| block1_pool (MaxPooling2D) | (None, 200, 200, 64)  | 0       |
| block2_conv1 (Conv2D)      | (None, 200, 200, 128) | 73856   |
| block2_conv2 (Conv2D)      | (None, 200, 200, 128) | 147584  |
| block3_conv3 (Conv2D)      | (None, 50, 50, 512)   | 1180160 |
| block2_pool (MaxPooling2D) | (None, 100, 100, 128) | 0       |
| block3_conv1 (Conv2D)      | (None, 100, 100, 256) | 295168  |
| block3_conv2 (Conv2D)      | (None, 100, 100, 256) | 590080  |
| block3_conv3 (Conv2D)      | (None, 100, 100, 256) | 590080  |
| block3_pool (MaxPooling2D) | (None, 50, 50, 256)   | 0       |
| block4_conv1 (Conv2D)      | (None, 50, 50, 512)   | 1180160 |
| block4_conv2 (Conv2D)      | (None, 50, 50, 512)   | 2359808 |
| block4_conv3 (Conv2D)      | (None, 50, 50, 512)   | 2359808 |
| block4_pool (MaxPooling2D) | (None, 25, 25, 512)   | 0       |
| block5_conv1 (Conv2D)      | (None, 25, 25, 512)   | 2359808 |
| block5_conv2 (Conv2D)      | (None, 25, 25, 512)   | 2359808 |
| block5_conv3 (Conv2D)      | (None, 25, 25, 512)   | 2359808 |
| block5_pool (MaxPooling2D) | (None, 12, 12, 512)   | 0       |
| flatten (Flatten)          | (None, 73728)         | 0       |
| dense (Dense)              | (None, 128)           | 9437312 |
| dense_1 (Dense)            | (None, 128)           | 16512   |
| dense_2 (Dense)            | (None, 64)            | 8256    |
| dense_3 (Dense)            | (None, 5)             | 325     |



__Total params: 24,177,093__\
__Trainable params: 9,462,405__\
__Non-trainable params: 14,714,688__



## Configuraion
------

### Data configuration



```yaml
"data_set_conf": {
    "annotations_path": "car_plates_dataset/annotations",
    "images_path": "car_plates_dataset/images",
    "data_dimension": 400
}
```


__annotations_path__: path to xml files of the dataset.

__images_path__:   path to images files of the dataset.

__data_dimension__: image dimensions, will be changed according to this input (I used 400x400).

### Training configuration


```yaml
"training_conf": {
    "enable_training": true,
    "validation_ratio": 0.03,
    "epoch_num": 25,
    "batch_size": 10,
    "object_existence_loss_weight": 0.5,
    "bounding_box_loss_weight": 5,
    "optimizer_learning_rate": 0.00001
}

```

__enable_training__: enables training, if on false will search for a saved model in the projects directory.

__validation_ratio__: divides the data set into validation and training.

__epoch_num__: number of epoch for training.

__batch_size__: batch size for every epoch. splits the data.

__object_existence_loss_weight__: object loss weight.

__bounding_box_loss_weight__: bounding box regression loss weight.

__optimizer_learning_rate__: adam learning rate.


### Model Configuration

```yaml
"model_conf": { 
    "model_type": "VGG16",
    "model_top_config": {
      "Flatten_Layer": {
        "initializing_parameters": {},
        "layer_type": "Flatten"
      },
      "dense_128_1": {
        "initializing_parameters": {
          "units": 128,
          "activation": "relu"
        },
        "input_layer": "Flatten_Layer",
        "layer_type": "Dense"
      },
      "dense_128_2": {
        "initializing_parameters": {
          "units": 128,
          "activation": "relu"
        },
        "input_layer": "dense_128_1",
        "layer_type": "Dense"
      },
      "dense_64": {
        "initializing_parameters": {
          "units": 64,
          "activation": "relu"
        },
        "input_layer": "dense_128_2",
        "layer_type": "Dense"
      },
      "dense_1_existence": {
        "initializing_parameters": {
          "units": 5,
          "activation": "sigmoid"
        },
        "input_layer": "dense_64",
        "layer_type": "Dense"
      }
    }
  },
```
  
__model_type__: type of the model's back bone - supports VGG16,VGG19 and Resnet50.

__model_top_config__: model's top configuration for fine tuning.

### Callback Configuration

```yaml
"call_back_conf": {
    "ReduceLROnPlateau": {
      "patience": 5,
      "verbose": 1,
      "factor": 0.00005
    }
  }

```

__ReduceLROnPlateau__: indicates the type of callback. LearningRateScheduler, ModelCheckpoint and EarlyStopping are also supported. the rest of the parameters are just initialized class members.

## Results
------

![Training Loss Graph](https://github.com/OrElimelech/computer_vision_projects/blob/main/car_plate_localization/results/training%20loss%20Graph.jpg "Training Loss Graph")
![Validation Loss Graph](https://github.com/OrElimelech/computer_vision_projects/blob/main/car_plate_localization/results/validation%20val_loss%20Graph.jpg "Validation Loss Graph")
![Plate Detection Example - Plate in Image](https://github.com/OrElimelech/computer_vision_projects/blob/main/car_plate_localization/results/result5.JPG "Plate Detection Example - Plate in Image")
![Plate Detection Example - No Plate in Image](https://github.com/OrElimelech/computer_vision_projects/blob/main/car_plate_localization/results/result7.JPG "Plate Detection Example - No Plate in Image")


## Notes
------

__it seems that the detection part of the classifier works quite well but the bounding box detection can be improved by using anchor bounding box from the used data set (with K-mean) and IoU calculations__.

__Those are used in the Yolo architecture. Those methods will be explored in other projects__.

__please check result folder to see additional outcome examples__.

__You can also add your own imges by just putting them in the test_images folder__.
