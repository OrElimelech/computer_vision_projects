# Hand-Written Digits Recognition


## Description
------
This project utilizes Tensorflow's pre-trained classifiers (I used VGG16) to detect handwrittern digits.
This uses Tensorflow's mnist data set.
I also made a virtual drawer to write digits and test the predictions of the trained model.
The outcome can be seen in the results folder.



 

## Model Structure
------



| Layer (type)               | Output Shape        | Param   |
|----------------------------|---------------------|---------|
| input_1 (InputLayer)       | [(None, 32, 32, 3)] | 0       |
| block1_conv1 (Conv2D)      | (None, 32, 32, 64)  | 1792    |
| block1_conv2 (Conv2D)      | (None, 32, 32, 64)  | 36928   |
| block1_pool (MaxPooling2D) | (None, 16, 16, 64)  | 0       |
| block2_conv1 (Conv2D)      | (None, 16, 16, 128) | 73856   |
| block2_conv2 (Conv2D)      | (None, 16, 16, 128) | 147584  |
| block2_pool (MaxPooling2D) | (None, 8, 8, 128)   | 0       |
| block3_conv1 (Conv2D)      | (None, 8, 8, 256)   | 295168  |
| block3_conv2 (Conv2D)      | (None, 8, 8, 256)   | 590080  |
| block3_conv3 (Conv2D)      | (None, 8, 8, 256)   | 590080  |
| block3_pool (MaxPooling2D) | (None, 4, 4, 256)   | 0       |
| block4_conv1 (Conv2D)      | (None, 4, 4, 512)   | 1180160 |
| block4_conv2 (Conv2D)      | (None, 4, 4, 512)   | 2359808 |
| block4_conv3 (Conv2D)      | (None, 4, 4, 512)   | 2359808 |
| block4_pool (MaxPooling2D) | (None, 2, 2, 512)   | 0       |
| block5_conv1 (Conv2D)      | (None, 2, 2, 512)   | 2359808 |
| block5_conv2 (Conv2D)      | (None, 2, 2, 512)   | 2359808 |
| block5_conv3 (Conv2D)      | (None, 2, 2, 512)   | 2359808 |
| block5_pool (MaxPooling2D) | (None, 1, 1, 512)   | 0       |
| flatten (Flatten)          | (None, 512)         | 0       |
| dense (Dense)              | (None, 1000)        | 513000  |
| dense_1 (Dense)            | (None, 10)          | 10010   |




__Total params: 15,237,698__\
__Trainable params: 523,010__\
__Non-trainable params: 14,714,688__





## Configuraion
------

### Training configuration



```yaml
"training_config": {
    "enable_training": true,
    "epochs": 25,
    "batch_size": 32,
    "call_backs_conf": {
      "EarlyStopping": {
        "patience": 5,
        "verbose": 1
      }
    }
  },
```


__enable_training__: enables the training of the model.

__epochs__:   number of training epochs.

__batch_size__: image batch size.

__call_back_conf__: call back configuration forr training.



### Data configuration


```yaml
"data_config": {
    "image_shape": {
      "image_rows": 32,
      "image_columns": 32,
      "channel_numbers": 3
    },
    "data_augmentation": {
      "rotation_range": 10
    }
  },

```

__image_shape__: shape of the image - rows,columns and number of channels.

__data_augmentation__: this projects uses Tensorflow's default generators that enables data augmentation. I used rotation only. more can be used by adding them to the configuration.




### Model Configuration

```yaml
"model_configuration": {
    "model_name": "VGG16",
    "model_top_config": {
      "Flatten_Layer": {
        "initializing_parameters": {},
        "layer_type": "Flatten"
      },
      "dense_1": {
        "initializing_parameters": {
          "units": 1000,
          "activation": "relu"
        },
        "input_layer": "Flatten_Layer",
        "layer_type": "Dense"
      },
      "dense_2": {
        "initializing_parameters": {
          "units": 10,
          "activation": "softmax"
        },
        "input_layer": "dense_1",
        "layer_type": "Dense"
      }
    }
  }
```
  
__model_type__: type of the model's back bone - supports VGG16,VGG19 and Resnet50.

__model_top_config__: model's top configuration for fine tuning.


## Results
------

![Training Confuision Matrix](https://github.com/OrElimelech/computer_vision_projects/blob/main/handwritten_digits_recognition/results/Confusion_Matrix_Training.jpg.jpg "Training Confuision Matrix")
![Validation Confuision Matrix](https://github.com/OrElimelech/computer_vision_projects/blob/main/handwritten_digits_recognition/results/Confusion_Matrix_Validation.jpg.jpg "Validation Confuision Matrix")
![Training Loss Graph](https://github.com/OrElimelech/computer_vision_projects/blob/main/handwritten_digits_recognition/results/training%20loss%20Graph.jpg "Training Loss Graph")
![Validation Loss Graph](https://github.com/OrElimelech/computer_vision_projects/blob/main/handwritten_digits_recognition/results/validation%20val_loss%20Graph.jpg "Validation Loss Graph")
![Training Accuracy Graph](https://github.com/OrElimelech/computer_vision_projects/blob/main/handwritten_digits_recognition/results/training%20accuracy%20Graph.jpg "Training Accuracy Graph")
![Validation Accuracy Graph](https://github.com/OrElimelech/computer_vision_projects/blob/main/handwritten_digits_recognition/results/validation%20val_accuracy%20Graph.jpg "Validation Accuracy Graph")
![Result Sample-1](https://github.com/OrElimelech/computer_vision_projects/blob/main/handwritten_digits_recognition/results/result1.JPG "Result Sample-1")
![Result Sample-2](https://github.com/OrElimelech/computer_vision_projects/blob/main/handwritten_digits_recognition/results/result2.JPG "Result Sample-2")
![Result Sample-3](https://github.com/OrElimelech/computer_vision_projects/blob/main/handwritten_digits_recognition/results/result3.JPG "Result Sample-3")


## Notes
------

__it seems that some digits can be problematic. I ran into issues with the 5 digit, it probably needs more '5' samples__.

_for more result samples please check the result folder.
