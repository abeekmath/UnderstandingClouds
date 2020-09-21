# Understanding Clouds From Satellite Images

Implemented a UNet From Scratch for Cloud Image Segmenation.

## Table of Contents:

- [UnderstandingClouds](#understandingclouds-pytorch)
    - [Data](#data)
    - [Model](#model)
    - [Experiment configs](#experiment-configs)
    - [Requirements](#requirements)
    - [Usage](#usage)
    - [Results](#results)
    - [References](#references)
    - [License](#license)


### Data:
- Implementation based on the Max-Planck Dataset for clouds.

### Model:
- Implemented Unet and DeepLabV3. Open-sourced Unet implementation. 
- DeepLabV3 Implementation based on torchvision models. 


### Experiment configs:
```
- Input size: 350 x 525 x 3
- Batch size: 4
- Learning rate: 5e-4
- Optimizer: Adam
- Model was trained locally on a GTX 1050Ti. 
```


### Usage:
- Clone the repository [here](https://github.com/abhirupkamath/UnderstandingClouds)
- ``` python main.py ```
- Runs on a GPU, you need to enable cuda in the config file.

### Results:
Achieved a Dice Coefficient of 0.59

### Requirements:
Check [requirements.txt](https://github.com/abhirupkamath/UnderstandingClouds/blob/master/requirements.txt).

### References:
- RLE Decoding : https://bit.ly/3ceBpCq
- Convex Hull Creation and Post Processing: https://bit.ly/3hNXVDj


### License:
This project is licensed under Apache 2.0 LICENSE - see the LICENSE file for details.
