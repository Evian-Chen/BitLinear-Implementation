# BitLinear: Implementation of BitNet's Quantized Linear Layer

This repository contains an implementation of the `BitLinear` module based on the paper [BitNet](https://arxiv.org/pdf/2310.11453), which introduces a quantized linear layer for efficient deep learning models.

## Features
- **Weight Quantization:** Converts weights to binary values using a custom `Sign` function.
- **Activation Quantization:** Supports both linear and non-linear (e.g., ReLU) activations with adjustable bitwidth.
- **Layer Normalization:** Applies normalization before quantization for stable training.
- **Dequantization:** Ensures outputs can be interpreted correctly by scaling results back.

## Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/bitlinear.git
cd bitlinear
