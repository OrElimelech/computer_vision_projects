# Nuclei Segmentation


## Description
------
This project utilizes the U-net capabilities to create mask images from input images of cells.\
The U-net blocks number can be changed and altered for experimentation



used data set: "https://www.kaggle.com/espsiyam/nuclei-image-segmentation"
 

## Model Structure
------



| Layer (type)                                                                                       | Output Shape               | Param #             | Connected to          |
|----------------------------------------------------------------------------------------------------|----------------------------|---------------------|-----------------------|
| input_1 (InputLayer)                                                                               | [(None, 256, 256, 3) 0     |                     |                       |
| conv2d (Conv2D)                                                                                    | (None, 256, 256, 16) 448   | input_1[0][0]       |                       |
| dropout (Dropout)                                                                                  | (None, 256, 256, 16) 0     | conv2d[0][0]        |                       |
| conv2d_1 (Conv2D)                                                                                  | (None, 256, 256, 16) 2320  | dropout[0][0]       |                       |
| max_pooling2d (MaxPooling2D)                                                                       | (None, 128, 128, 16) 0     | conv2d_1[0][0]      |                       |
| conv2d_2 (Conv2D)                                                                                  | (None, 128, 128, 32) 4640  | max_pooling2d[0][0] |                       |
| dropout_1 (Dropout)                                                                                | (None, 128, 128, 32) 0     | conv2d_2[0][0]      |                       |
| conv2d_3 (Conv2D)                                                                                  | (None, 128, 128, 32) 9248  | dropout_1[0][0]     |                       |
| max_pooling2d_1 (MaxPooling2D)                                                                     | (None, 64, 64, 32)         | 0                   | conv2d_3[0][0]        |
| conv2d_4 (Conv2D)                                                                                  | (None, 64, 64, 64)         | 18496               | max_pooling2d_1[0][0] |
| dropout_2 (Dropout)                                                                                | (None, 64, 64, 64)         | 0                   | conv2d_4[0][0]        |
| conv2d_5 (Conv2D)                                                                                  | (None, 64, 64, 64)         | 36928               | dropout_2[0][0]       |
| max_pooling2d_2 (MaxPooling2D)                                                                     | (None, 32, 32, 64)         | 0                   | conv2d_5[0][0]        |
| conv2d_6 (Conv2D)                                                                                  | (None, 32, 32, 128)        | 73856               | max_pooling2d_2[0][0] |
| dropout_3 (Dropout)                                                                                | (None, 32, 32, 128)        | 0                   | conv2d_6[0][0]        |
| conv2d_7 (Conv2D)                                                                                  | (None, 32, 32, 128)        | 147584              | dropout_3[0][0]       |
| max_pooling2d_3 (MaxPooling2D)                                                                     | (None, 16, 16, 128)        | 0                   | conv2d_7[0][0]        |
| conv2d_8 (Conv2D)                                                                                  | (None, 16, 16, 256)        | 295168              | max_pooling2d_3[0][0] |
| dropout_4 (Dropout)                                                                                | (None, 16, 16, 256)        | 0                   | conv2d_8[0][0]        |
| conv2d_9 (Conv2D)                                                                                  | (None, 16, 16, 256)        | 590080              | dropout_4[0][0]       |
| max_pooling2d_4 (MaxPooling2D)                                                                     | (None, 8, 8, 256)          | 0                   | conv2d_9[0][0]        |
| conv2d_10 (Conv2D)                                                                                 | (None, 8, 8, 512)          | 1180160             | max_pooling2d_4[0][0] |
| dropout_5 (Dropout)                                                                                | (None, 8, 8, 512)          | 0                   | conv2d_10[0][0]       |
| conv2d_11 (Conv2D)                                                                                 | (None, 8, 8, 512)          | 2359808             | dropout_5[0][0]       |
| conv2d_transpose (Conv2DTranspo (None, 16, 16, 256)                                                | 524544                     | conv2d_11[0][0]     |                       |
| concatenate (Concatenate)                                                                          | (None, 16, 16, 512)        | 0                   | conv2d_9[0][0]        |
| conv2d_transpose[0][0]                                                                             |                            |                     |                       |
| conv2d_12 (Conv2D)                                                                                 | (None, 16, 16, 256)        | 1179904             | concatenate[0][0]     |
| dropout_6 (Dropout)                                                                                | (None, 16, 16, 256)        | 0                   | conv2d_12[0][0]       |
| conv2d_13 (Conv2D)                                                                                 | (None, 16, 16, 256)        | 590080              | dropout_6[0][0]       |
| conv2d_transpose_1 (Conv2DTrans (None, 32, 32, 128)                                                | 131200                     | conv2d_13[0][0]     |                       |
| concatenate_1 (Concatenate)                                                                        | (None, 32, 32, 256)        | 0                   | conv2d_7[0][0]        |
| conv2d_transpose_1[0][0]                                                                           |                            |                     |                       |
| conv2d_14 (Conv2D)                                                                                 | (None, 32, 32, 128)        | 295040              | concatenate_1[0][0]   |
| dropout_7 (Dropout)                                                                                | (None, 32, 32, 128)        | 0                   | conv2d_14[0][0]       |
| conv2d_15 (Conv2D)                                                                                 | (None, 32, 32, 128)        | 147584              | dropout_7[0][0]       |
| conv2d_transpose_2 (Conv2DTrans (None, 64, 64, 64)                                                 | 32832                      | conv2d_15[0][0]     |                       |
| concatenate_2 (Concatenate)                                                                        | (None, 64, 64, 128)        | 0                   | conv2d_5[0][0]        |
| conv2d_transpose_2[0][0]                                                                           |                            |                     |                       |
| conv2d_16 (Conv2D)                                                                                 | (None, 64, 64, 64)         | 73792               | concatenate_2[0][0]   |
| dropout_8 (Dropout)                                                                                | (None, 64, 64, 64)         | 0                   | conv2d_16[0][0]       |
| conv2d_17 (Conv2D)                                                                                 | (None, 64, 64, 64)         | 36928               | dropout_8[0][0]       |
| conv2d_transpose_3 (Conv2DTrans (None, 128, 128, 32) 8224                                          | conv2d_17[0][0]            |                     |                       |
| concatenate_3 (Concatenate)                                                                        | (None, 128, 128, 64) 0     | conv2d_3[0][0]      |                       |
| conv2d_transpose_3[0][0]                                                                           |                            |                     |                       |
| conv2d_18 (Conv2D)                                                                                 | (None, 128, 128, 32) 18464 | concatenate_3[0][0] |                       |
| dropout_9 (Dropout)                                                                                | (None, 128, 128, 32) 0     | conv2d_18[0][0]     |                       |
| conv2d_19 (Conv2D)                                                                                 | (None, 128, 128, 32) 9248  | dropout_9[0][0]     |                       |
| conv2d_transpose_4 (Conv2DTrans (None, 256, 256, 16) 2064                                          | conv2d_19[0][0]            |                     |                       |
| concatenate_4 (Concatenate)                                                                        | (None, 256, 256, 32) 0     | conv2d_1[0][0]      |                       |
| conv2d_transpose_4[0][0]                                                                           |                            |                     |                       |
| conv2d_20 (Conv2D)                                                                                 | (None, 256, 256, 16) 4624  | concatenate_4[0][0] |                       |
| dropout_10 (Dropout)                                                                               | (None, 256, 256, 16) 0     | conv2d_20[0][0]     |                       |
| conv2d_21 (Conv2D)                                                                                 | (None, 256, 256, 16) 2320  | dropout_10[0][0]    |                       |
| conv2d_22 (Conv2D)                                                                                 | (None, 256, 256, 1)        | 17                  | conv2d_21[0][0]       |


__Total params: 7,775,601__\
__Trainable params: 7,775,601__\
__Non-trainable params: 0__


## Configuraion
------

### Module configuration



```yaml
"model_structure": {
    "encoder_params": {
      "general_conv_params": {
        "kernel_size": [
          3,
          3
        ],
        "activation": "elu",
        "padding": "same",
        "kernel_initializer": "he_normal"
      },
      "max_pool_size": [
        2,
        2
      ],
      "changing_conv_params": {
        "encoder_block_1": {
          "dropout_rate": 0.1,
          "filter_size": 16
        },
        "encoder_block_2": {
          "dropout_rate": 0.1,
          "filter_size": 32
        },
        "encoder_block_3": {
          "dropout_rate": 0.2,
          "filter_size": 64
        },
        "encoder_block_4": {
          "dropout_rate": 0.2,
          "filter_size": 128
        },
        "encoder_block_5": {
          "dropout_rate": 0.3,
          "filter_size": 256
        }
      }
    },
    "bottle_neck_params": {
      "general_conv_params": {
        "kernel_size": [
          3,
          3
        ],
        "activation": "elu",
        "padding": "same",
        "kernel_initializer": "he_normal"
      },
      "changing_conv_params": {
        "dropout_rate": 0.3,
        "filter_size": 512
      }
    },
    "decoder_params": {
      "general_conv_params": {
        "kernel_size": [
          3,
          3
        ],
        "activation": "elu",
        "padding": "same",
        "kernel_initializer": "he_normal"
      },
      "general_transpose_conv_params": {
        "kernel_size": [
          2,
          2
        ],
        "padding": "same",
        "strides": [
          2,
          2
        ]
      },
      "changing_conv_params": {
        "decoder_block_1": {
          "dropout_rate": 0.3,
          "filter_size": 256,
          "concatenate_layer": "encoder_block_5"
        },
        "decoder_block_2": {
          "dropout_rate": 0.2,
          "filter_size": 128,
          "concatenate_layer": "encoder_block_4"
        },
        "decoder_block_3": {
          "dropout_rate": 0.2,
          "filter_size": 64,
          "concatenate_layer": "encoder_block_3"
        },
        "decoder_block_4": {
          "dropout_rate": 0.1,
          "filter_size": 32,
          "concatenate_layer": "encoder_block_2"
        },
        "decoder_block_5": {
          "dropout_rate": 0.1,
          "filter_size": 16,
          "concatenate_layer": "encoder_block_1"
        }
      }
    }
  }
```


__encoder_params__: configuration of the encoder part of the model.

__general_conv_params__: general parameters for each convolution block (encoder or decoder).

__changing_conv_params__: changing parameters for each convolution block (encoder or decoder).

__decoder_params__: configuration of the decoder part of the model.

__call_backs_conf__: configuration for Tensorflow's default callbacks instances.

__general_transpose_conv_params__: general parameters for each de-convolution block (decoder only).





### Data configuration


```yaml
"data_set_location": "U_NET",

```

__data_set_location__: path to data set.



## Results
------

![example 1](https://github.com/OrElimelech/computer_vision_projects/blob/main/nuclei_segmentation/results/result2.JPG "example 1")

![example 2](https://github.com/OrElimelech/computer_vision_projects/blob/main/nuclei_segmentation/results/result4.JPG "example 2")

![example 3](https://github.com/OrElimelech/computer_vision_projects/blob/main/nuclei_segmentation/results/result7.JPG "example 3")



## Notes
------

__it seems that there is a data set biass - there are more histogram images than reguler 3 channels images, thst is why the mask is reversed on those kind of images__.

