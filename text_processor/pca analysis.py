import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle

# Load the embeddings from a file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/biobertLEMembeddings_biochem.pkl', 'rb') as f:
    embeddings_biochem = pickle.load(f)

with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/biobertLEMembeddings_biomech.pkl', 'rb') as f:
    embeddings_biomech = pickle.load(f)

# Combine the embeddings from both fields into one array
embeddings = np.concatenate((embeddings_biochem, embeddings_biomech), axis=0)

# Fit a PCA model to the data and obtain the explained variance
pca = PCA()
pca.fit(embeddings)
explained_variance = pca.explained_variance_ratio_

# Plot the cumulative sum of the explained variance as a function of the number of components
cumulative_variance = np.cumsum(explained_variance)
plt.plot(cumulative_variance)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

# Calculate cumulative explained variance
cumulative_var = np.cumsum(pca.explained_variance_ratio_)

# Find the number of components that explain 90% of the variance
n_components = np.argmax(cumulative_var >= 0.90) + 1
print(f"Number of components that explain 90% of variance: {n_components}")