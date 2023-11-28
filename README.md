# PIXIENN

A modern C++ reimplementation of Darknet with CUDA support for efficient neural network inference.

![Inference Result 1](resources/examples/predictions.jpg)

## Overview

This project aims to provide a modern and optimized C++ implementation of the Darknet neural network framework, focusing on efficient inference with CUDA support. The goal is to achieve improved performance compared to the original Darknet while maintaining accuracy.

## Features

- **CUDA Support:** Utilize the power of NVIDIA GPUs for accelerated neural network inference.
- **Model Compatibility:** Successfully tested with YOLOv1-tiny, YOLOv3-tiny, and YOLOv3 models.
- **Performance:** Significantly reduced inference time compared to the original Darknet implementation.
- **Modern C++ Features:** Leveraging modern C++ standards for improved readability and maintainability.

## Getting Started

### Prerequisites

- C++ Compiler
- OpenBLAS library
- Boost (>= 1.74)
- OpenCV (>= 4.5.4)
- LibTIFF
- nlohmann_json (>= 3.10.5)
- yaml-cpp
- CUDA Toolkit (for GPU support)
- CUDNN8 (for GPU support)

### Installation

```bash
# Clone the repository
git clone https://github.com/trieck/pixienn

# Build the project
cd pixienn
mkdir build
cd build
cmake ..
make
