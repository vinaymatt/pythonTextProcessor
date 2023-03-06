from gensim.models import Word2Vec, Phrases
from transformers import AutoTokenizer
import re
import nltk
import pandas as pd
from gensim.models import Word2Vec

# Load BioBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

# Add special tokens
special_tokens_dict = {'additional_special_tokens': ['mechanotransduction', 'mechanobiology']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# Define the file path
file_path = '/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/AllAbstracts.txt'

# Read the file
with open(file_path, 'r') as file:
    corpus = file.read()

# Tokenize the corpus with BioBERT tokenizer
max_length = 512
sequences = nltk.sent_tokenize(corpus)
tokens = []
for seq in sequences:
    if len(seq) > max_length:
        subseqs = [seq[i:i+max_length] for i in range(0, len(seq), max_length)]
        tokens.extend([tokenizer.encode(subseq) for subseq in subseqs])
    else:
        tokens.append(tokenizer.encode(seq))

# Convert tokens to list of strings
tokens = [[tokenizer.decode(t) for t in seq] for seq in tokens]

# Remove stopwords and non-alphabetic characters
stopwords = nltk.corpus.stopwords.words('english')
corpus_cleaned = []
for seq in tokens:
    seq_cleaned = [re.sub(r'[^a-zA-Z]', '', word).lower() for word in seq if word not in stopwords]
    corpus_cleaned.append(seq_cleaned)

# Build word pairs
window_size = 8
bigram_transformer = Phrases(corpus_cleaned, min_count=5, threshold=10)
corpus_bigrams = [bigram_transformer[seq] for seq in corpus_cleaned]
pairs = []
for seq in corpus_bigrams:
    for i, word in enumerate(seq):
        for j in range(max(0, i - window_size), min(len(seq), i + window_size + 1)):
            if i != j:
                pairs.append((word, seq[j]))

# Train Word2Vec model
model = Word2Vec(corpus_bigrams, vector_size=200, window=8, min_count=4, workers=4)

# create dictionary of similar words for each word in vocab
sim_dict = {}
for word in model.wv.index_to_key:
    sim_words = model.wv.most_similar(word, topn=10)
    sim_dict[word] = sim_words

# write to excel file
output = pd.DataFrame(columns=['Word', 'Similar Word 1', 'Cosine Similarity 1', 'Similar Word 2', 'Cosine Similarity 2',
                               'Similar Word 3', 'Cosine Similarity 3', 'Similar Word 4', 'Cosine Similarity 4',
                               'Similar Word 5', 'Cosine Similarity 5', 'Similar Word 6', 'Cosine Similarity 6',
                               'Similar Word 7', 'Cosine Similarity 7', 'Similar Word 8', 'Cosine Similarity 8',
                               'Similar Word 9', 'Cosine Similarity 9', 'Similar Word 10', 'Cosine Similarity 10'])
for word in model.wv.key_to_index:
    sim_words = sim_dict[word]
    row = [word]
    for i in range(10):
        sim_word, sim_cosine = sim_words[i]
        row.extend([sim_word, sim_cosine])
    output.loc[len(output)] = row

# set formatting for excel file
writer = pd.ExcelWriter('word_similarity.xlsx', engine='xlsxwriter')
output.to_excel(writer, index=False)
workbook = writer.book
worksheet = writer.sheets['Sheet1']
bold = workbook.add_format({'bold': True})
worksheet.set_column('A:A', None, bold)
worksheet.set_column('B:U', 20)
writer.save()
