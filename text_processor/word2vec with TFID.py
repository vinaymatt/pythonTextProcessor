import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

# Load the preprocessed biochemistry abstracts file
with open(
        '/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge/preprocessedword2vec_abstracts_biochem.txt',
        'r') as f:
    biochem_contents = f.read()
    biochem_abstracts = biochem_contents.split('\n\n')

# Load the preprocessed biomechanics abstracts file
with open(
        '/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/preprocessedword2vec_abstracts_biomech.txt',
        'r') as f:
    biomech_contents = f.read()
    biomech_abstracts = biomech_contents.split('\n\n')

# Load the original biochemistry abstracts file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge/Abstracts.txt', 'r') as f:
    biochem_original = f.read()
    biochem_original_abstracts = biochem_original.split('\n\n')

# Load the original biomechanics abstracts file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/AllAbstracts.txt', 'r') as f:
    biomech_original = f.read()
    biomech_original_abstracts = biomech_original.split('\n\n')

# Tokenize biochemistry and biomechanics abstracts into sentences and words
tokenized_biochem_abstracts = [simple_preprocess(abstract) for abstract in biochem_abstracts]
tokenized_biomech_abstracts = [simple_preprocess(abstract) for abstract in biomech_abstracts]

# Combine both sets of abstracts for training the Word2Vec model
all_abstracts = tokenized_biochem_abstracts + tokenized_biomech_abstracts

# Train a Word2Vec model (skip-gram version) on the tokenized abstracts
model = Word2Vec(all_abstracts, sg=1, vector_size=300, window=5, min_count=5, workers=4, epochs=10)

# Compute TF-IDF values for words in the abstracts
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, min_df=5)
tfidf_matrix = tfidf_vectorizer.fit_transform(all_abstracts)

# Create a dictionary to map words to their corresponding TF-IDF values
word2tfidf = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))

# Function to create an embedding for each abstract by averaging the embeddings of its words, weighted by their TF-IDF values
def abstract_embedding(abstract, model, word2tfidf):
    valid_words = [word for word in abstract if word in model.wv.key_to_index and word in word2tfidf]
    if valid_words:
        embeddings = model.wv[valid_words]
        weights = np.array([word2tfidf[word] for word in valid_words])
        normalized_weights = weights / weights.sum()
        return np.dot(normalized_weights, embeddings)
    else:
        return None


# Create weighted embeddings for biochemistry and biomechanics abstracts
biochem_embeddings = [abstract_embedding(abstract, model, word2tfidf) for abstract in tokenized_biochem_abstracts]
biomech_embeddings = [abstract_embedding(abstract, model, word2tfidf) for abstract in tokenized_biomech_abstracts]

# Filter out abstracts for which embeddings couldn't be generated
biochem_embeddings = [embedding for embedding in biochem_embeddings if embedding is not None]
biomech_embeddings = [embedding for embedding in biomech_embeddings if embedding is not None]

# Filter out abstracts for which embeddings couldn't be generated
filtered_biochem_abstracts = [abstract for abstract, embedding in zip(biochem_abstracts, biochem_embeddings) if embedding is not None]
filtered_biomech_abstracts = [abstract for abstract, embedding in zip(biomech_abstracts, biomech_embeddings) if embedding is not None]

# Save biochemistry embeddings to a pickle file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge/word2vec_withTFID_biochem_embeddings.pkl', 'wb') as f:
    pickle.dump(biochem_embeddings, f)

print(len(biochem_embeddings))

# Save biomechanics embeddings to a pickle file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/word2vec_withTFID_biomech_embeddings.pkl', 'wb') as f:
    pickle.dump(biomech_embeddings, f)

print(len(biomech_embeddings))

# Save filtered biochemistry abstracts to a new file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge/preprocessedword2vec_withTFID_abstracts_biochem.txt', 'w') as f:
    f.write('\n\n'.join(filtered_biochem_abstracts))

print(len(filtered_biochem_abstracts))

# Save filtered biomechanics abstracts to a new file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/preprocessedword2vec_withTFID_abstracts_biomech.txt', 'w') as f:
    f.write('\n\n'.join(filtered_biomech_abstracts))

print(len(filtered_biomech_abstracts))

# Filter out original biochemistry abstracts for which embeddings couldn't be generated
filtered_biochem_original_abstracts = [abstract for abstract, embedding in zip(biochem_original_abstracts, biochem_embeddings) if embedding is not None]

# Filter out original biomechanics abstracts for which embeddings couldn't be generated
filtered_biomech_original_abstracts = [abstract for abstract, embedding in zip(biomech_original_abstracts, biomech_embeddings) if embedding is not None]

# Save filtered original biochemistry abstracts to a new file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge/word2vec_withTFID_original_filtered_abstracts_biochem.txt', 'w') as f:
    f.write('\n\n'.join(filtered_biochem_original_abstracts))

print(len(filtered_biochem_original_abstracts))

# Save filtered original biomechanics abstracts to a new file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/word2vec_withTFID_original_filtered_abstracts_biomech.txt', 'w') as f:
    f.write('\n\n'.join(filtered_biomech_original_abstracts))

print(len(filtered_biomech_original_abstracts))
