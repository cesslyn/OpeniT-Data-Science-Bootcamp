# Transformer Concepts - Multi-Head Attention Lab
import numpy as np

# Key Shapes
np.random.seed(11)
batch_size = 2
sequence_length = 4
model_dim = 8
num_heads = 2
head_dim = model_dim // num_heads
assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
X = np.random.normal(size=(batch_size, sequence_length, model_dim))
print("X shape:", X.shape)
print("head_dim:", head_dim)

# Softmax and Attention Logic
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2)
    weights = softmax(scores / np.sqrt(d_k), axis=-1)
    output = weights @ V
    return output, weights

# Projection Matrices
W_q = np.random.normal(size=(model_dim, model_dim))
W_k = np.random.normal(size=(model_dim, model_dim))
W_v = np.random.normal(size=(model_dim, model_dim))
W_o = np.random.normal(size=(model_dim, model_dim))
Q = X @ W_q
K = X @ W_k
V = X @ W_v
print("Q before split:", Q.shape)

# Split Into Heads
def split_heads(x, batch_size, num_heads, head_dim):
    x = x.reshape(batch_size, sequence_length, num_heads, head_dim)
    return np.transpose(x, (0, 2, 1, 3))
Q_heads = split_heads(Q, batch_size, num_heads, head_dim)
K_heads = split_heads(K, batch_size, num_heads, head_dim)
V_heads = split_heads(V, batch_size, num_heads, head_dim)
print("Q after split:", Q_heads.shape)
print("K after split:", K_heads.shape)
print("V after split:", V_heads.shape)

print('(1) What does each axis in Q_heads represent?')
print('ANSWER: Each axis in Q_heads represents batch_size, num_heads, and head_dim.\n')
print('(2) Why does each head receive only head_dim values instead of model_dim values?')
print('ANSWER: The model dimension is divided among multiple heads that is why each head receive only head_dim values instead of model_dim values.\n')
print('(3) Why does sequence_length stay unchanged?')
print('ANSWER: The sequence_length stay unchanged because attention is computed per token, not across tokens as a reduction.\n')

head_outputs, head_weights = scaled_dot_product_attention(Q_heads, K_heads, V_heads)
print("Head outputs shape:", head_outputs.shape)
print("Head attention weights shape:", head_weights.shape)

# Interpretation
print("\nInterpretation:")
print("head_outputs shape =")
print("(batch_size, num_heads, sequence_length, head_dim)")

print("\nhead_weights shape =")
print("(batch_size, num_heads, sequence_length, sequence_length)")

print("\nEach attention head learns its own attention pattern.")

# Heads and Project Output
def combine_heads(x, batch_size, sequence_length, model_dim):
    x = np.transpose(x, (0, 2, 1, 3))
    return x.reshape(batch_size, sequence_length, model_dim)
combined = combine_heads(head_outputs, batch_size, sequence_length, model_dim)
output = combined @ W_o
print("Combined heads shape:", combined.shape)
print("Final output shape:", output.shape)

# Head Comparison
for head_index in range(num_heads):
    print(f"Attention weights for sample 0, head {head_index}:")
    print(head_weights[0, head_index])

print('\n(1) Do the heads produce identical weights?')
print('ANSWER: No, each head produces different attention weights because each head has different learned projection matrices (Wq, Wk, Wv), allowing them to focus on different relationships in the sequence.\n')
print('(2) Why might one head focus on nearby tokens while another focuses on different tokens?')
print('ANSWER: Different attention heads learn different relationships in the sequence.One head may specialize in local context (nearby words), while another may capture long-range dependencies, grammar, or semantic meaning.\n')
print('(3) Why is the final output projected with W_o after heads are combined?')
print('ANSWER: After concatenating all attention heads, the projection matrix W_o mixes information from all heads into a single representation.\n')
