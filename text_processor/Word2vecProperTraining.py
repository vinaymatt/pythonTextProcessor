import re
import pickle
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from collections import defaultdict

class TqdmWord2Vec(Word2Vec):
    def train(self, *args, **kwargs):
        kwargs['total_examples'] = kwargs['total_examples'] or self.corpus_count
        with tqdm(total=kwargs['total_examples'], desc="Training Word2Vec") as pbar:
            for _ in range(self.epochs):
                super().train(*args, **kwargs)
                pbar.update(kwargs['total_examples'] // self.epochs)
# Read abstracts
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/Temp/preprocessedword2vec_abstracts_biomech.txt', 'r') as f:
    abstracts = f.read().split('\n\n')

# Tokenize abstracts
tokenized_abstracts = [re.findall(r"[\wα-ωΑ-Ω'-]+", abstract) for abstract in tqdm(abstracts, desc="Tokenizing abstracts")]

# Read special phrases
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/cleanedproteinforword2vecfinal.txt', 'r') as f:
    special_phrases_raw = f.read().splitlines()
special_phrases = [tuple(phrase.split()) for phrase in special_phrases_raw]

# Find and annotate bigrams/trigrams using Gensim's Phrases and Phraser
phrases = Phrases(tokenized_abstracts, min_count=1, threshold=1)
bigram_phraser = Phraser(phrases)
tokenized_abstracts_bigram = [bigram_phraser[abstract] for abstract in tqdm(tokenized_abstracts, desc="Annotating bigrams/trigrams")]

# Search for special phrases (case-insensitive, whole words only)
special_phrases_contexts = defaultdict(list)
for phrase in tqdm(special_phrases, desc="Searching special phrases"):
    phrase_regex = r'\b' + r'\s+'.join(re.escape(word) for word in phrase) + r'\b'
    pattern = re.compile(phrase_regex, flags=re.IGNORECASE)
    for abstract in tokenized_abstracts_bigram:
        abstract_text = ' '.join(abstract)
        for match in pattern.finditer(abstract_text):
            start_idx = len(abstract_text[:match.start()].split())
            end_idx = len(abstract_text[:match.end()].split())
            context = abstract_text[max(start_idx - 10, 0):start_idx] + abstract_text[end_idx:min(end_idx + 10, len(abstract))]
            special_phrases_contexts[match.group().lower().replace(' ', '_')].append(context)

# Train Word2Vec model
model = TqdmWord2Vec(sentences=tokenized_abstracts_bigram, sg=1, vector_size=200, window=9, min_count=5, workers=4, epochs=17)  # Use skip-gram

# Add special phrases to the model and train with their surrounding context
for phrase, contexts in tqdm(special_phrases_contexts.items(), desc="Training special phrases"):
    if phrase not in model.wv.key_to_index:
        model.build_vocab([phrase], update=True)
        model.train(contexts, total_examples=len(contexts), epochs=model.epochs)

# Save embeddings to a pkl file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/Temp/wordembeddings3.pkl', 'wb') as f:
    pickle.dump(model.wv, f)

# Save tokenized_abstracts_bigram to a pickle file
with open("/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/Temp/tokenized_abstracts_bigram.pkl", "wb") as f:
    pickle.dump(tokenized_abstracts_bigram, f)

