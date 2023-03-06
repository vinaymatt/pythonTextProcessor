import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from umap import UMAP
import numpy as np

# Load the embeddings from the pickle file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge/word2vec_withTFID_biochem_embeddings.pkl', 'rb') as f:
    biochem_embeddings = pickle.load(f)

with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/word2vec_withTFID_biomech_embeddings.pkl', 'rb') as f:
    biomech_embeddings = pickle.load(f)

# Combine the embeddings
biochem_count = len(biochem_embeddings)
combined_embeddings = np.vstack((biochem_embeddings, biomech_embeddings))

# Perform dimensionality reduction for 2D visualization using UMAP
reduced_embeddings_2d = UMAP(n_neighbors=30, n_components=2).fit_transform(combined_embeddings)

# Separate the 2D embeddings for biochemistry and biomechanics abstracts
biochem_embeddings_2d = reduced_embeddings_2d[:biochem_count]
biomech_embeddings_2d = reduced_embeddings_2d[biochem_count:]

# Set up the plot
sns.set(style="white")
plt.figure(figsize=(12, 8))

# Plot the 2D embeddings with different colors for biochemistry and biomechanics abstracts
plt.scatter(biochem_embeddings_2d[:, 0], biochem_embeddings_2d[:, 1], label='Biochemistry', c='red', alpha=0.5)
plt.scatter(biomech_embeddings_2d[:, 0], biomech_embeddings_2d[:, 1], label='Biomechanics', c='blue', alpha=0.5)

# Add legend and labels
plt.legend()
plt.xlabel('UMAP 1st Component')
plt.ylabel('UMAP 2nd Component')
plt.title('UMAP Visualization of Biochemistry and Biomechanics Abstracts')

# Save the plot to a file
plt.savefig('umap_visualization.png', dpi=300)

# Show the plot
plt.show()
