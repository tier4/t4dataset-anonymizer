# t4dataset-anonymizer: High-Efficiency, Real-Time Anonymization Pipeline for Edge AI in the T4 Dataset

The **t4dataset-anonymizer** enables real-time image anonymization for the T4 dataset.  
This pipeline uses **trt-lightnet** <sup>[[1]](#trt-lightnet)</sup>.

https://github.com/user-attachments/assets/5831e723-7fc6-4501-8b2c-ac13a95ece86

---

## Installation

### Requirements

#### Supported Platforms & Toolchains

| Platform | CUDA | TensorRT | OS |
|---|---|---|---|
| x86_64 (desktop/server) | 11.0 – 12.6 | **8.5 / 8.6 / 10.x** | Ubuntu 22.04 |
| Jetson (JetPack 5.x) | 11.4 | 8.6.x | Ubuntu 20.04 (L4T) |
| Jetson (JetPack 6.x) | 12.2+ | **10.x** | Ubuntu 22.04 (L4T) |

> Notes  
> - TensorRT **8.x** and **10.x** both work. The build auto-detects your installed version and switches the appropriate code path.  
> - Prebuilt `.engine` files are **not guaranteed** to be compatible across major TRT versions or different GPU architectures. Rebuild engines after upgrading TensorRT, CUDA, or GPU.

#### Dependencies

- `libgflags-dev`
- `libboost-all-dev`
- `libopencv-dev`
- **cnpy** (for tensor debug) – install from https://github.com/rogersce/cnpy

This repository has been tested with the following environments:

- CUDA 11.7 + TensorRT 8.5.2 on Ubuntu 22.04  
- CUDA 12.2 + TensorRT 8.6.0 on Ubuntu 22.04  
- CUDA 11.4 + TensorRT 8.6.0 on Jetson JetPack 5.1  
- CUDA 11.8 + TensorRT 8.6.1 on Ubuntu 22.04  
- **CUDA 12.8 + TensorRT 10.8 on Ubuntu 22.04**  

### Steps for Local Installation

1) Clone the repository

```bash
git clone git@github.com:tier4/t4dataset-anonymizer.git
cd t4dataset-anonymizer
```

2) Install libraries

```bash
sudo apt update
sudo apt install -y libgflags-dev libboost-all-dev libopencv-dev
# Install cnpy from source (required)
# https://github.com/rogersce/cnpy
```

3) Build

Basic build (auto-detects TRT 8.x vs 10.x):

```bash
mkdir -p build && cd build
cmake ..
make -j"$(nproc)"
```

If CMake cannot find TensorRT automatically, set one (or more) of:

```bash
# Example paths — adjust to your environment
export TensorRT_ROOT=/usr/lib/x86_64-linux-gnu
export TensorRT_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvinfer.so
export TensorRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu

# Then:
cmake -DTensorRT_ROOT="$TensorRT_ROOT" -DTensorRT_INCLUDE_DIR="$TensorRT_INCLUDE_DIR" ..
make -j"$(nproc)"
```

#### Jetson Tips

- **JP 5.x (TRT 8.6.x):** Use system TensorRT and CUDA 11.4.  
- **JP 6.x (TRT 10.x):** Use system TensorRT and CUDA 12.x. Ensure `CUDNN` is present (installed with JetPack).  
- Swap file recommended for large ONNX → TRT conversion.

---

## Model
T.B.D

---

## Usage

### Converting anonymization models to TensorRT engines

Build TRT engine:

```bash
./t4dataset-anonymizer --flagfile ../configs/CONFIGS.txt
```

### Inference with the TensorRT engine

From a directory of images:

```bash
./t4dataset-anonymizer --flagfile ../configs/CONFIGS.txt --d DIRECTORY
```

From a video file:

```bash
./t4dataset-anonymizer --flagfile ../configs/CONFIGS.txt --v VIDEO
```

From a **T4 dataset** and write anonymized data into `{T4_DATASET_NAME}/anonymized_data/`:

```bash
./t4dataset-anonymizer --flagfile ../configs/CONFIGS.txt --dont_show --t4d {T4_DATASET_NAME}
```

---

## TensorRT 10 Notes (Read Before Upgrading)

- **API changes:** TRT 10 removes/changes some legacy APIs used in older samples. This project uses version guards and will:
  - Use `builder->buildSerializedNetwork(*network, *config)` and `runtime->deserializeCudaEngine(...)` on TRT ≥ 8.0 (incl. **10.x**).
  - Avoid deprecated `buildEngineWithConfig` code paths on TRT 10.
- **ONNX Parser:** Make sure your `nvonnxparser` in TensorRT 10 matches your ONNX opset. Re-export ONNX if conversion fails.
- **Engine portability:** Rebuild engines after upgrades or hardware changes (e.g., moving between L40S ↔ H100, or 8.x ↔ 10.x).
- **Precision:** If you previously used FP16/INT8, confirm calibration/calib cache paths in `CONFIGS.txt` (INT8) and re-generate cache after major upgrades.

Example of guarded C++ code path used internally (for reference only):

```cpp
#if (NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSOR_PATCH) >= 8000
  auto plan = TrtUniquePtr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create host memory");
    return false;
  }
  runtime_->setEngineHostCodeAllowed(true);
  engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(
      runtime_->deserializeCudaEngine(plan->data(), plan->size()));
#else
  engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(
      builder->buildEngineWithConfig(*network, *config));
#endif
```

---

## Troubleshooting

- **Cannot find TensorRT:** export `TensorRT_ROOT`, `TensorRT_INCLUDE_DIR`, or `TensorRT_LIBRARY` and re-run CMake.  
- **Engine build fails after upgrade:** clean `build/`, re-export ONNX, and rebuild.  
- **INT8 mismatch or accuracy drop:** delete old calibration cache, re-run INT8 calibration on the target device.  
- **Jetson memory issues:** close desktop GUI, increase swap, and reduce max workspace size in builder config.

---

## References
[1]. <a id="trt-lightnet"></a> [trt-lightnet](https://github.com/tier4/trt-lightnet)

---

## (Optional) CONFIGS.txt Hints

Common flags you may want in `configs/CONFIGS.txt`:

```
# IO
input=path/to/input
output=path/to/output

# Precision (set depending on your target)
precision=fp16
# precision=int8
# calib_cache=./calibration.cache

# Engine build
max_workspace_size_mb=4096
opt_batch=1
max_batch=1

# Logging
verbose=0
```

> For TRT 10, `max_workspace_size_mb` still applies; tune per GPU memory budget.
