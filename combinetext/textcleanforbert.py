import re
import nltk
from transformers import AutoTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tqdm


# Read the abstracts from a txt file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge/Abstracts.txt', 'r') as f:
    contents = f.read()
    abstracts = contents.split('\n\n')

print(len(abstracts))

# Initialize the BioBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Convert all abstracts to lowercase
abstracts = [abstract.lower() for abstract in abstracts]

# Remove non-alphabetic characters using regular expressions
abstracts = [re.sub(r'[^a-zA-Z0-9\-\/\(\)\[\]]', ' ', abstract) for abstract in abstracts]

# Tokenize the abstracts using the BioBERT tokenizer
abstracts = [tokenizer.tokenize(abstract) for abstract in abstracts]

# Perform named entity recognition (NER) and replace with standard labels
abstracts_ner = []
for abstract in tqdm.tqdm(abstracts):
    ne_tree = nltk.ne_chunk(nltk.pos_tag(abstract))
    ne_sent = []
    for chunk in ne_tree:
        if hasattr(chunk, 'label'):
            ne_sent.append(chunk.label() + '_ENTITY')
        else:
            ne_sent.append(chunk[0])
    abstracts_ner.append(ne_sent)
abstracts = abstracts_ner

# Remove stop words using a more specialized list of stop words
stop_words = set(stopwords.words('english') + ['abstract', 'background', 'conclusion', 'objective'])
abstracts = [[word for word in tqdm.tqdm(abstract) if word not in stop_words] for abstract in abstracts]

# Lemmatize the remaining words using the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()
abstracts = [[lemmatizer.lemmatize(word) for word in tqdm.tqdm(abstract)] for abstract in abstracts]

# Filter out non-noun and non-verb words using POS tagging
abstracts_pos = []
for abstract in tqdm.tqdm(abstracts):
    pos_tags = nltk.pos_tag(abstract)
    pos_sent = []
    for word, tag in pos_tags:
        if tag.startswith('N') or tag.startswith('V'):
            pos_sent.append(word)
    abstracts_pos.append(pos_sent)
abstracts = abstracts_pos

# Normalize numerical and chemical entities
abstracts_norm = []
for abstract in tqdm.tqdm(abstracts):
    norm_sent = []
    for word in abstract:
        if re.match(r'^-?\d+(?:\.\d+)?(?:e-?\d+)?$', word):
            # Normalize numerical entities
            norm_sent.append(str(float(word)))
        elif re.match(r'^[A-Z0-9]+$', word):
            # Normalize chemical entities
            norm_sent.append(word.lower())
        else:
            norm_sent.append(word)
    abstracts_norm.append(norm_sent)
abstracts = abstracts_norm

# Join the words back into sentences
abstracts = [' '.join(abstract) for abstract in abstracts]

print(len(abstracts))

# Save the preprocessed abstracts to a new file
with open('/data/Biochemlarge/preprocessedword2vec_abstracts_biochem.txt', 'w') as f:
    f.write('\n\n'.join(abstracts))
