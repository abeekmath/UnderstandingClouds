# Understanding Clouds From Satellite Images

Implemented a UNet From Scratch for Cloud Image Segmenation.

## Table of Contents:

- [UnderstandingClouds]
    - [Data](#data)
    - [Model](#model)
    - [Experiment configs](#experiment-configs)
    - [Requirements](#requirements)
    - [Usage](#usage)
    - [Results](#results)
    - [References](#references)
    - [License](#license)


### Data:
 

### Model:


### Experiment configs:
```
- Input size: 224 x 224 x 3
- Batch size: 16
- Learning rate: 1e-4
- Optimizer: Adam
```
### Usage:
- Clone the repository [here](https://github.com/abhirupkamath/DeepPixFace/blob/master/config.py)
- ``` python main.py ```
- Runs on a GPU, you need to enable cuda in the config file.

### Results:
Achieved a Dice Coefficient of 0.59

### Requirements:
Check [requirements.txt](https://github.com/abhirupkamath/DeepPixFace/blob/master/requirements.txt).


### License:
This project is licensed under Apache 2.0 LICENSE - see the LICENSE file for details.
