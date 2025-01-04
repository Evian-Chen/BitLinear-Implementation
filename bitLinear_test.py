from BitLinear import BitLinear
import torch
import torch.nn as nn

def test_initialization():
    bit_linear = BitLinear(in_features=4, out_features=2)
    assert bit_linear.weight.shape == (2, 4), "Weight shape mismatch"
    assert isinstance(bit_linear.layer_norm, nn.LayerNorm), "LayerNorm not initialized correctly"
    print("Initialization test passed!")

def test_forward_pass():
    bit_linear = BitLinear(in_features=4, out_features=2)
    x = torch.randn(3, 4)  # [batch_size=3, in_features=4]
    output = bit_linear(x)
    assert output.shape == (3, 2), f"Output shape mismatch: {output.shape}"
    print("Forward pass test passed!")

def test_weight_quantization():
    bit_linear = BitLinear(in_features=4, out_features=2)
    x = torch.randn(3, 4)  # [batch_size=3, in_features=4]
    _ = bit_linear(x)  
    unique_weights = torch.unique(bit_linear.weight)
    assert torch.all((unique_weights == 1.0) | (unique_weights == -1.0)), "Weight quantization failed"
    print("Weight quantization test passed!")

def test_activation_quantization():
    bit_linear = BitLinear(in_features=4, out_features=2)
    x = torch.randn(3, 4)  # [batch_size=3, in_features=4]
    Qb = 2 ** (bit_linear.bits - 1)
    gama = bit_linear.absmax(x)
    x_quantized = torch.clamp(x * (Qb / gama), min=-Qb + bit_linear.epsilon, max=Qb - bit_linear.epsilon)

    assert torch.all(x_quantized >= -Qb + bit_linear.epsilon), "Activation quantization min range failed"
    assert torch.all(x_quantized <= Qb - bit_linear.epsilon), "Activation quantization max range failed"
    print("Activation quantization test passed!")

def test_output_range():
    bit_linear = BitLinear(in_features=4, out_features=2)
    x = torch.randn(3, 4)  # [batch_size=3, in_features=4]
    output = bit_linear(x)

    assert torch.isfinite(output).all(), "Output contains NaN or Inf"
    print("Output range test passed!")

test_initialization()
test_forward_pass()
test_weight_quantization()
test_activation_quantization()
test_output_range()