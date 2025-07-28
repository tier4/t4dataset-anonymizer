# t4dataset-anonymizer: High-Efficiency, Real-Time Anonymization Pipeline for Edge AI in the T4 Dataset

The "t4dataset-anonymizer" enables real-time image anonymization for the T4 dataset.
This pipeline uses "trt-lightnet <sup>[[1]](#trt-lightnet)</sup>".



https://github.com/user-attachments/assets/5831e723-7fc6-4501-8b2c-ac13a95ece86



## Installation

### Requirements

#### For Local Installation

-   CUDA 11.0 or later
-   TensorRT 8.5 or 8.6
-   cnpy for debug of tensors
This repository has been tested with the following environments:

- CUDA 11.7 + TensorRT 8.5.2 on Ubuntu 22.04
- CUDA 12.2 + TensorRT 8.6.0 on Ubuntu 22.04
- CUDA 11.4 + TensorRT 8.6.0 on Jetson JetPack5.1
- CUDA 11.8 + TensorRT 8.6.1 on Ubuntu 22.04

### Steps for Local Installation

1.  Clone the repository.

```shell
$ git clone git@github.com:tier4/t4dataset-anonymizer.git
$ cd t4dataset-anonymizer
```

2.  Install libraries.

```shell
$ sudo apt update
$ sudo apt install libgflags-dev
$ sudo apt install libboost-all-dev
$ sudo apt install libopencv-dev
```

Install from the following repository.

https://github.com/rogersce/cnpy


3.  Compile the TensorRT implementation.

```shell
$ mkdir build && cd build
$ cmake ../
$ make -j
```

## Model
 T.B.D

## Usage

### Converting anonymization models to TensorRT engines

Build TRT engine
```shell
$ ./t4dataset-anonymizer --flagfile ../configs/CONFIGS.txt 
```
### Inference with the TensorRT engine

Inference from images
```shell
$ ./t4dataset-anonymizer --flagfile ../configs/CONFIGS.txt  --d DIRECTORY
```

Inference from images
```shell
$ ./t4dataset-anonymizer --flagfile ../configs/CONFIGS.txt  --v VIDEO
```

Inference from t4dataset and generate anonymized data into "{T4_DATASET_NAME}/anonymized_data/"
```shell
$ ./t4dataset-anonymizer --flagfile ../configs/CONFIGS.txt --dont_show --t4d {T4_DATASET_NAME}
```


# References
[1]. [trt-lightnet](https://github.com/tier4/trt-lightnet)  
