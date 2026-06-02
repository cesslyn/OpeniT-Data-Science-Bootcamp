# Optional: transformer_utils.py
import torch
import torch.nn as nn
class TinyMultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        assert model_dim % num_heads == 0
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.output = nn.Linear(model_dim, model_dim)
    def forward(self, x):
        batch_size, sequence_length, model_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        def split_heads(tensor):
            tensor = tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim)
            return tensor.transpose(1, 2)
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        scores = q @ k.transpose(-2, -1)
        weights = torch.softmax(scores / (self.head_dim ** 0.5), dim=-1)
        context = weights @ v
        context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, model_dim)
        return self.output(context), weights
    
if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.randn(2, 4, 8)
    layer = TinyMultiHeadSelfAttention(model_dim=8, num_heads=2)
    output, weights = layer(x)
    print("Output shape:", output.shape)
    print("Weights shape:", weights.shape)

print('\n(1) Which parts of the PyTorch code match your NumPy implementation?')
print('ANSWER: The PyTorch code matches the NumPy implementation in creating Q, K, and V, computing attention scores, applying softmax, and combining the values.\n')

print('(2) Why does the qkv layer output 3 * model_dim values?')
print('ANSWER: The qkv layer outputs 3 * model_dim values because it generates the query, key, and value vectors at the same time.\n')

print('(3) Which tensor contains the attention probabilities?')
print('ANSWER: The tensor `weights` contains the attention probabilities after the softmax operation.\n')