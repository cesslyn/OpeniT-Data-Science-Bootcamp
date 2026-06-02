# Transformer Concepts - Positional Encoding Lab
import numpy as np
import matplotlib.pyplot as plt

# Problem Position Solves
print('(1) What information is missing if a model only sees token embeddings?')
print('ANSWER: The information that is missing if the model only sees token embeddings is the order of position of the words or tokens. The token embeddings only represent meaning, not where each word appears.\n')
print('(2) Why are the sequences ["data", "science", "bootcamp"] and ["bootcamp", "science", "data"] different even if they contain the same tokens?')
print('ANSWER: The sequences ["data", "science", "bootcamp"] and ["bootcamp", "science", "data"] are different even if they contain the same tokens because the word order changes the meaning and context of the sequence.\n')
print('(3) What should positional encoding add to each token vector?')
print("ANSWER: Positional encoding should add information about the token's positiona or index in the sequence to each token vestor.")

# Small Embedding Matrix
sequence_length = 4
embedding_dim = 6
np.random.seed(42)
token_embeddings = np.random.normal(size=(sequence_length, embedding_dim))
print("Token embeddings shape:", token_embeddings.shape)
print(token_embeddings)

# Sinusoidal Positional Encoding
def sinusoidal_positional_encoding(sequence_length, embedding_dim):
    positions = np.arange(sequence_length)[:, np.newaxis]
    dimensions = np.arange(embedding_dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / embedding_dim)
    angle_radians = positions * angle_rates
    positional_encoding = np.zeros((sequence_length, embedding_dim))
    positional_encoding[:, 0::2] = np.sin(angle_radians[:, 0::2])
    positional_encoding[:, 1::2] = np.cos(angle_radians[:, 1::2])
    return positional_encoding
position_vectors = sinusoidal_positional_encoding(sequence_length, embedding_dim)
print("Position encoding shape:", position_vectors.shape)
print(position_vectors)

# Token and Position Vectors
input_vectors = token_embeddings + position_vectors
print("Input vectors shape:", input_vectors.shape)
print(input_vectors)

print('\n(1) Why do token_embeddings and position_vectors need the same shape?')
print('ANSWER: The token_embeddings and position_vestors need the same shape so that they can be added element-wise. Each token embedding must match its corresponding positional encoding vector.\n')
print('(2) What does each row represent after the addition?')
print('ANSWER: Each row represents a token with both its semantic meaning and its position information combined.\n')
print('(3) What does each column represent after the addition?')
print('ANSWER: Each column represents onme feature or dimension of the combined embedding and positional information.\n')

# Visualization of Positional Encoding
plt.figure(figsize=(8, 4))
plt.imshow(position_vectors, aspect="auto")
plt.colorbar(label="Encoding value")
plt.xlabel("Embedding dimension")
plt.ylabel("Token position")
plt.title("Sinusoidal Positional Encoding")
plt.tight_layout()
plt.savefig("positional_encoding_heatmap.png")
print("Saved chart: positional_encoding_heatmap.png")
