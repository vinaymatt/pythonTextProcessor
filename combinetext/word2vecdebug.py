import re
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from collections import defaultdict
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Read abstracts
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/Temp/preprocessedword2vec_abstracts_biomech.txt', 'r') as f:
    abstracts = f.read().split('\n\n')

# Tokenize abstracts
tokenized_abstracts = [re.findall(r"[\wα-ωΑ-Ω-']+|[a-zA-Z]+-?[a-zA-Z]+", abstract) for abstract in
                       tqdm(abstracts, desc="Tokenizing abstracts")]

# Read special phrases
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/cleanedproteinforword2vecfinal.txt', 'r') as f:
    special_phrases_raw = f.read().splitlines()
special_phrases = [tuple(phrase.replace(" ", "_").split()) for phrase in special_phrases_raw]

# Find and annotate bigrams using Gensim's Phrases and Phraser
bigram_phrases = Phrases(tokenized_abstracts, min_count=1, threshold=1, scoring='npmi')
bigram_phraser = Phraser(bigram_phrases)
tokenized_abstracts_bigram = [bigram_phraser[abstract] for abstract in tqdm(tokenized_abstracts, desc="Annotating bigrams")]

# Find and annotate trigrams using Gensim's Phrases and Phraser
trigram_phrases = Phrases(tokenized_abstracts_bigram, min_count=1, threshold=1, scoring='npmi')
trigram_phraser = Phraser(trigram_phrases)
tokenized_abstracts_bigram_trigram = [trigram_phraser[bigram_abstract] for bigram_abstract in tqdm(tokenized_abstracts_bigram, desc="Annotating trigrams")]

# Prepare DataFrame
columns = ['special_phrase', 'match', 'left_context', 'right_context', 'match_found']
df = pd.DataFrame(columns=columns)

# Create a dictionary to store special phrase matches
matches = defaultdict(list)
encountered_phrases = set()

# Search for special phrases (case-insensitive, whole words only)
for abstract in tqdm(tokenized_abstracts_bigram_trigram, desc="Searching special phrases"):
    encountered_phrases = set()
    for i in range(len(abstract)):
        for phrase in special_phrases:
            phrase_len = len(phrase)
            if tuple(abstract[i:i + phrase_len]) == phrase and phrase not in encountered_phrases:
                start_idx = max(i - 10, 0)
                end_idx = min(i + phrase_len + 10, len(abstract))
                left_context = abstract[start_idx:i]
                right_context = abstract[i + phrase_len:end_idx]
                match_found = True
                match = ' '.join(phrase)

                row_data = {
                    'special_phrase': ' '.join(phrase),
                    'match': match,
                    'left_context': ' '.join(left_context),
                    'right_context': ' '.join(right_context),
                    'match_found': bool(match_found)  # Explicitly cast to bool dtype
                }

                df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
                encountered_phrases.add(phrase)

        # Add unmatched special_phrases
        for phrase in set(special_phrases) - encountered_phrases:
            row_data = {
                'special_phrase': ' '.join(phrase),
                'match': '',
                'left_context': '',
                'right_context': '',
                'match_found': False
            }

            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

# Save DataFrame to Excel file
df.to_excel('/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/special_phrases_contexts4.xlsx', index=False)
