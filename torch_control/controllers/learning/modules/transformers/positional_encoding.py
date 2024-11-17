import math
from enum import Enum
import torch
import torch.nn as nn


def get_pos_encoding(name, d_model, max_len=5000):
    if name == 'abs':
        return AbsolutePositionalEncoding(d_model, max_len)
    elif name == 'rel':
        return RelativePositionalEncoding(d_model, max_len)
    elif name == 'learnable':
        return LearnablePositionalEncoding(d_model, max_len)
    elif name == 'rotary':
        return RotaryPositionalEncoding(d_model, max_len)
    else:
        raise ValueError("Unknown positional encoding type: {}".format(name))

# 1. Absolute Positional Encoding (Sine and Cosine Functions)
class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(AbsolutePositionalEncoding, self).__init__()
        # Create a long enough 'pe' matrix with size (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register 'pe' as a buffer that should not be considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(0), :]


# 2. Relative Positional Encoding
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.rel_positions = torch.arange(-max_len, max_len + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        all_terms = torch.zeros(max_len * 2 + 1, d_model)
        all_terms[:, 0::2] = torch.sin(self.rel_positions * div_term)
        all_terms[:, 1::2] = torch.cos(self.rel_positions * div_term)
        self.register_buffer('sin_cos_terms', all_terms)

    def forward(self, q_pos, k_pos):
        rel_positions = k_pos - q_pos[:, None]
        clipped_rel_positions = torch.clamp(rel_positions, -self.max_len, self.max_len)
        index = clipped_rel_positions + self.max_len
        return self.sin_cos_terms[index]


# 3. Learnable Positional Encoding
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(max_len, 1, d_model))

    def forward(self, x):
        return self.pe[:x.size(0), :]
    
# 4. Rotary Positional Encoding
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be an even number for RotaryPositionalEncoding")
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        t = torch.arange(max_len).type_as(inv_freq)
        freqs = t[:, None] * inv_freq[None, :]
        self.register_buffer('sin', freqs.sin())
        self.register_buffer('cos', freqs.cos())

    def rotate_every_two(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        sin, cos = self.sin[:x.size(0), None, :], self.cos[:x.size(0), None, :]
        return torch.cat((cos * x1 - sin * x2, sin * x1 + cos * x2), dim=-1).reshape_as(x)

    def forward(self, x):
        return self.rotate_every_two(x)

    
################# Test #################
def test_absolute_positional_encoding():
    d_model = 512
    max_len = 100
    ape = AbsolutePositionalEncoding(d_model, max_len)

    # Create a dummy input tensor of size [sequence_length, batch_size, d_model]
    seq_len = 10
    batch_size = 1
    dummy_input = torch.zeros(seq_len, batch_size, d_model)

    # Get the positional encoding for the dummy input
    positional_encoding = ape(dummy_input)

    # Check the shape of the positional encoding
    assert positional_encoding.shape == (seq_len, 1, d_model), "Shape mismatch"

    # Check specific value properties
    # For example, compare the first two positions' sine and cosine values
    position_0 = positional_encoding[0, 0, :]
    position_1 = positional_encoding[1, 0, :]
    
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    expected_pos_0 = torch.zeros(d_model)
    expected_pos_0[0::2] = torch.sin(torch.tensor(0.0) * div_term)
    expected_pos_0[1::2] = torch.cos(torch.tensor(0.0) * div_term)
    expected_pos_1 = torch.zeros(d_model)
    expected_pos_1[0::2] = torch.sin(torch.tensor(1.0) * div_term)
    expected_pos_1[1::2] = torch.cos(torch.tensor(1.0) * div_term)

    assert torch.allclose(position_0, expected_pos_0, atol=1e-5), "Position 0 encoding mismatch"
    assert torch.allclose(position_1, expected_pos_1, atol=1e-5), "Position 1 encoding mismatch"

    print("Absolute Positional Encoding tests passed.")

def test_relative_positional_encoding():
    d_model = 512
    max_len = 100
    rpe = RelativePositionalEncoding(d_model, max_len)

    q_pos = torch.tensor([0, 1])
    k_pos = torch.tensor([0, 1, 2])

    positional_encoding = rpe(q_pos, k_pos)
    expected_shape = (q_pos.size(0), k_pos.size(0), d_model)
    assert positional_encoding.shape == expected_shape, "Shape mismatch"

    # Check specific values
    # For example, comparing the encoding at relative position 0 (q_pos = 0, k_pos = 0)
    relative_pos_0 = positional_encoding[0, 0, :]
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    expected_encoding = torch.zeros(d_model)
    expected_encoding[0::2] = torch.sin(torch.tensor(0.0) * div_term)
    expected_encoding[1::2] = torch.cos(torch.tensor(0.0) * div_term)

    assert torch.allclose(relative_pos_0, expected_encoding, atol=1e-5), "Relative position 0 encoding mismatch"

    print("Relative Positional Encoding tests passed.")

def test_learnable_positional_encoding():
    d_model = 512
    max_len = 100
    lpe = LearnablePositionalEncoding(d_model, max_len)

    seq_len = 10
    batch_size = 1
    dummy_input = torch.zeros(seq_len, batch_size, d_model)

    positional_encoding_before = lpe(dummy_input).clone()

    # Simulate a simple training step
    optimizer = torch.optim.SGD(lpe.parameters(), lr=0.1)
    loss = positional_encoding_before.sum()
    loss.backward()
    optimizer.step()

    positional_encoding_after = lpe(dummy_input)

    # Check if the encoding has changed
    assert not torch.allclose(positional_encoding_before, positional_encoding_after, atol=1e-5), "Positional encodings should be updated after training step"

    print("Learnable Positional Encoding tests passed.")
    
def test_rotary_positional_encoding():
    d_model = 512  # Should be even
    max_len = 100
    rope = RotaryPositionalEncoding(d_model, max_len)

    seq_len = 10
    batch_size = 1
    dummy_input = torch.randn(seq_len, batch_size, d_model)

    encoded_output = rope(dummy_input)

    # Check the shape of the output
    assert encoded_output.shape == dummy_input.shape, "Shape mismatch"

    # Check if the rotation preserves the norm (since it's a rotation)
    norm_before = torch.norm(dummy_input, dim=-1)
    norm_after = torch.norm(encoded_output, dim=-1)

    assert torch.allclose(norm_before, norm_after, atol=1e-5), "Rotation should preserve the norm"

    print("Rotary Positional Encoding tests passed.")

    

if __name__ == "__main__":
    print("Running tests...")
    
    print("Test 1: Absolute Positional Encoding")
    # Run the test
    test_absolute_positional_encoding()
    
    print("Test 2: Relative Positional Encoding")
    # Run the test
    test_relative_positional_encoding()
    
    print("Test 3: Learnable Positional Encoding")
    # Run the test
    test_learnable_positional_encoding()
    
    print("Test 4: Rotary Positional Encoding")
    # Run the test
    test_rotary_positional_encoding()
