import numpy as np

# Mock data for testing
N = 1099
D = 256

# Generate random embeddings (normalized)
embeddings = np.random.randn(N, D)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # unit norm

# Generate boundary distances: mix of correct/incorrect predictions
probabilities = np.random.beta(2, 2, N)  # skewed towards 0.5
true_labels = np.random.randint(0, 2, N)
boundary_distances = np.where(
    true_labels == 1, 2 * (probabilities - 0.5), -2 * (probabilities - 0.5)
)

# Save
np.save("embeddings.npy", embeddings)
np.save("boundary_distances.npy", boundary_distances)

print("Mock data generated.")
