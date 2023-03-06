import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm
import spacy
import os
import sys



# Read the protein names from a txt file
#with open('/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/cleanedproteinforword2vecfinal.txt', 'r') as f:
 #   protein_names = set(line.strip() for line in f)

# Load the SciSpacy models
nlp1 = spacy.load('en_ner_jnlpba_md')
nlp2 = spacy.load('en_ner_bionlp13cg_md')

entities = set()

def is_mixed_case(word):
    uppercase_count = sum(1 for letter in word if letter.isupper())
    return uppercase_count / len(word) > 0.5 and uppercase_count / len(word) < 0.99

def is_abbreviation(word):
    return word.isupper() or (len(word) == 1 and word.isalpha() and word.isupper()) or is_mixed_case(word)

def selectively_lowercase(abstract):
    doc1 = nlp1(abstract)
    doc2 = nlp2(abstract)

    # Extract named entities using the first NER model
    for ent in doc1.ents:
        if ent.label_ in ['PROTEIN', 'GENE']:
            entities.add(ent.text)

    # Extract named entities using the second NER model
    for ent in doc2.ents:
        if ent.label_ in ['PROTEIN', 'GENE']:
            entities.add(ent.text)

    # Add abbreviations to the entities set
    words = abstract.split()
    for word in words:
        if is_abbreviation(word):
            entities.add(word)

    # Lowercase words that are not entities or abbreviations
    for i, word in enumerate(words):
        if word not in entities:
            words[i] = word.lower()
    return ' '.join(words)

def preprocess(file_path):
    with open(file_path, 'r') as f:
        contents = f.read()
        abstracts = contents.split('\n\n')

    # Selectively lowercase words
    abstracts = [selectively_lowercase(abstract) for abstract in tqdm(abstracts, desc='Lowercasing', leave=False)]

    # Remove non-alphabetic characters using regular expressions
    abstracts = [re.sub(r'[^a-zA-Z0-9α-ωΑ-Ω\/\(\)\[\]-]', ' ', abstract) for abstract in abstracts]

    # Tokenize the abstracts using NLTK's word_tokenize
    abstracts = [nltk.word_tokenize(abstract) for abstract in tqdm(abstracts, desc='Tokenizing', ncols=100)]

    # Remove stop words using a more specialized list of stop words
    stop_words = set(stopwords.words('english') + ['abstract', 'background', 'conclusion', 'objective', 'result', 'method', 'study', 'purpose', 'aim', 'introduction', 'objective'])
    abstracts = [[word for word in abstract if word not in stop_words] for abstract in tqdm(abstracts, desc='Removing stopwords', file=sys.stdout)]

    # Lemmatize the remaining words using the WordNet lemmatizer, preserving protein names
    lemmatizer = WordNetLemmatizer()
    abstracts = [[lemmatizer.lemmatize(word) if word not in entities else word for word in abstract] for abstract in tqdm(abstracts, desc='Lemmatizing', file=sys.stdout)]

    # Filter out non-noun and non-verb words using POS tagging, preserving protein names
    abstracts_pos = []
    for abstract in tqdm(abstracts, desc='POS tagging'):
        pos_tags = nltk.pos_tag(abstract)
        pos_sent = [word for word, tag in pos_tags if tag.startswith('N') or tag.startswith('V') or word in entities]
        abstracts_pos.append(pos_sent)
    abstracts = abstracts_pos

    # Join the words back into sentences
    abstracts = [' '.join(abstract) for abstract in abstracts]

    print(len(abstracts))
    return abstracts

# Directory containing the abstract files
abstracts_dir = '/Users/vinay/PycharmProjects/pythonTextProcessor/text_processor/English_Abstracts'
output_dir = '/Users/vinay/PycharmProjects/pythonTextProcessor/text_processor/PreProcessed English Abstracts'

# Iterate over the files in the directory
for file_name in tqdm(os.listdir(abstracts_dir), desc="Processing files"):
    if file_name.endswith('.txt'):
        input_file_path = os.path.join(abstracts_dir, file_name)
        output_file_name = f'preprocessed_{file_name}'
        output_file_path = os.path.join(output_dir, output_file_name)

        preprocessed_abstracts = preprocess(input_file_path)

        with open(output_file_path, 'w') as f:
            f.write('\n\n'.join(preprocessed_abstracts))
