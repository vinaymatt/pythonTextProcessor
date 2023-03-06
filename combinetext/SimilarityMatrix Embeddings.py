import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import pickle

# Load the embeddings from the pickle file
with open("/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/Temp/wordembeddings3.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Load tokenized_abstracts_bigram from the pickle file
with open("/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/Temp/tokenized_abstracts_bigram.pkl", "rb") as f:
    tokenized_abstracts_bigram = pickle.load(f)

# Load the embeddings from the gensim model
word_vectors = embeddings  # Assuming 'embeddings' is the model.wv from your code

# Select relevant words (top N most frequent)
N = 1000
word_counts = Counter([word for abstract in tokenized_abstracts_bigram for word in abstract])
top_words = [word for word, _ in word_counts.most_common(N) if word in word_vectors.key_to_index]
selected_embeddings = {word: word_vectors[word] for word in top_words}

# Compute similarity matrix using cosine similarity
embedding_matrix = np.array([vec for vec in selected_embeddings.values()])
similarity_matrix = cosine_similarity(embedding_matrix)

# Create adjacency matrix with a threshold
threshold = 0.5
adjacency_matrix = (similarity_matrix > threshold).astype(int)
np.fill_diagonal(adjacency_matrix, 0)

# Save the adjacency matrix as a numpy file
np.save("/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/Temp/adjacency_matrix.npy", adjacency_matrix)
