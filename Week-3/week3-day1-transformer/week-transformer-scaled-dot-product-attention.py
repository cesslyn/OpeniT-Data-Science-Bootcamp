# Transformer Concepts - Scaled Dot-Product Attention Lab
import numpy as np
import matplotlib.pyplot as plt

# Toy Token Vectors
np.random.seed(7)
sequence_length = 4
model_dim = 8
X = np.random.normal(size=(sequence_length, model_dim))
print("Input X shape:", X.shape)

# Query, Key, and Value Projections
W_q = np.random.normal(size=(model_dim, model_dim))
W_k = np.random.normal(size=(model_dim, model_dim))
W_v = np.random.normal(size=(model_dim, model_dim))
Q = X @ W_q
K = X @ W_k
V = X @ W_v
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

# Raw Attention Scores
raw_scores = Q @ K.T
print("Raw scores shape:", raw_scores.shape)
print(raw_scores)

print('(1)  Why is K transposed?')
print('ANSWER: K is transposed so the dimensions match for matrix multiplication between Q and K^T in order to compute attention scores.\n')
print('(2) Why is the score matrix sequence_length by sequence_length?')
print('ANSWER: Score matrix is square because every token is compared with every other token in the sequence.\n')
print('(3) What does row 0 of raw_scores represent?')
print('ANSWER: Row 0 or raw_scores represents how strongly the first token attends to all token attends to all tokens in the sequence, including itself.\n')

# Scores
d_k = K.shape[-1]
scaled_scores = raw_scores / np.sqrt(d_k)
print("Scaled scores shape:", scaled_scores.shape)
print(scaled_scores)

# Softmax
def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)
attention_weights = softmax(scaled_scores, axis=-1)
print("Attention weights shape:", attention_weights.shape)
print(attention_weights)
print("Row sums:", attention_weights.sum(axis=-1))

print('(1) Why should each row sum to 1?')
print('ANSWER: Each row sum to 1 because softmax converts the scores into probabilities.\n')
print('(2)  What does a larger attention weight mean?')
print('ANSWER: A larger attention weight means token pays more attention to that specific token because it is more relevant or important.\n')
print('What does attention weight near 0 mean?')
print('ANSWER: An attention weight near 0 means the token gives very little attention or impotance to that token.\n')

# Attention Output
attention_output = attention_weights @ V
print("Attention output shape:", attention_output.shape)
print(attention_output)

print('\n(1)  Why is V not transposed here?')
print('ANSWER: V is not transposed because the attention weights are already arranged to multiply correctly with V.\n')
print('(2) What does each output row represent?')
print('ANSWER: Each output row reoresents the updated representation of a token after considering information from all tokens in the sequence.\n')
print('(3)  How is attention different from simply averaging all token vectors?')
print('ANSWER: Attention gives different importance to tokens using weights, while averaging treats all tokens equally.\n')

# Attention Heatmap
token_labels = ["Token 1", "Token 2", "Token 3", "Token 4"]
plt.figure(figsize=(6, 5))
plt.imshow(attention_weights, aspect="auto")
plt.colorbar(label="Attention weight")
plt.xticks(range(sequence_length), token_labels, rotation=45)
plt.yticks(range(sequence_length), token_labels)
plt.xlabel("Key token")
plt.ylabel("Query token")
plt.title("Scaled Dot-Product Attention Weights")
plt.tight_layout()
plt.savefig("attention_heatmap.png")
print("Saved chart: attention_heatmap.png")
