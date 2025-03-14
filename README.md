# Breast Mammography Classification with GA-MIL and MCDO

This project implements **Gated Attention Multiple Instance Learning (GA-MIL)** combined with **Monte Carlo Dropout (MCDO)** for breast mammography classification.

## Overview

- **GA-MIL**: Predicts classification outcomes and generates attention maps highlighting important image features.
- **MCDO**: Enables the computation of uncertainty in both prediction probabilities and attention weights, providing uncertainty-aware attention maps.

During the inference, computation overhead is minimal since feature extraction using a Convolutional Neural Network (CNN) is performed only once. The multiple forward passes with dropout during inference are limited to the attention module and the final classification module.

**NOTE**: In MIL, each input image is divided into patches, which are processed independently by e.g. ResNet-18 model for feature extraction. Since, a batch size of 1 is used, a reshape from 5D to 4D tensor is required transforming dims in the following `(1, N, C, H, W)` to `(N, C, H, W)`. Since the number of patches `(N)` varies between images, `BatchNorm2d()` would compute inconsistent statistics across different samples, leading to unstable training. Also `BatchNorm2d()` assumes the patches are independent samples which is not true in this case.

## System Architecture

\#todo

## Metrics

\#todo table (performance, inference time, mcdo, no-mcdo)

## Visualization

\#todo

## References

\#todo

\#todo
