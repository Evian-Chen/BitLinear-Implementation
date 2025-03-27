# BitLinear: Implementation of BitNet's Quantized Linear Layer

This repository provides an implementation of **BitLinear**, a quantized linear layer inspired by the [BitNet paper](https://arxiv.org/pdf/2310.11453).

BitLinear performs both weight and activation quantization for efficient linear transformations, incorporating Layer Normalization and quantization-aware operations for performance and numerical stability.

---

## Paper Reference

> **BitNet: Training Binary Neural Networks with Constant Memory and FLOPs**  
> Zhang, Z. et al., 2023.  
> [arXiv:2310.11453](https://arxiv.org/pdf/2310.11453)

---

## Features

- Weight quantization using binary values `{-1, 1}`
- Quantization-aware forward pass with activation scaling
- Optional support for linear or non-linear layers
- Integrated Layer Normalization
- Simple and minimal PyTorch design
- Includes unit tests for:
  - Initialization
  - Forward pass
  - Weight and activation quantization
  - Output value sanity check

## BitLinear computation flow
<img src="https://github.com/user-attachments/assets/62478cc4-757e-4280-a57c-0033f950988c" alt="Description of image" style="width:50%;"/>
