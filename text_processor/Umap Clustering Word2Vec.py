import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from umap import UMAP
import hdbscan


# Load the embeddings from the pickle file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge/word2vec_withTFID_biochem_embeddings.pkl', 'rb') as f:
    biochem_embeddings = pickle.load(f)

with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/word2vec_withTFID_biomech_embeddings.pkl', 'rb') as f:
    biomech_embeddings = pickle.load(f)

# Combine the embeddings
biochem_count = len(biochem_embeddings)
combined_embeddings = np.vstack((biochem_embeddings, biomech_embeddings))

# Perform dimensionality reduction using UMAP
reduced_embeddings_2d = UMAP(n_neighbors=30, n_components=2, random_state=42).fit_transform(combined_embeddings)

# Perform clustering using HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=30)
cluster_labels = clusterer.fit_predict(reduced_embeddings_2d)

# Set up the plot
sns.set(style="white")
plt.figure(figsize=(12, 8))

# Plot the 2D embeddings with cluster labels as colors
scatter = plt.scatter(reduced_embeddings_2d[:, 0], reduced_embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)

# Add a colorbar to indicate cluster labels
colorbar = plt.colorbar(scatter)
colorbar.set_label('Cluster Label')

# Add labels
plt.xlabel('UMAP 1st Component')
plt.ylabel('UMAP 2nd Component')
plt.title('HDBSCAN Clustering of Biochemistry and Biomechanics Abstracts')

# Save the plot to a file
plt.savefig('hdbscan_clusteringwithtfid.png', dpi=300)

# Show the plot
plt.show()