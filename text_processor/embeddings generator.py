from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import os
import pickle

os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.set_num_threads(4)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# Load the BERT model
model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.1')

with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge/preprocessed_abstracts_biochem.txt', 'r') as f:
    text_biochem = f.read()
abstracts_biochem = text_biochem.split('\n\n')
abstracts_biochem = [a.strip() for a in abstracts_biochem if a.strip()]

with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/preprocessed_abstracts_biomech.txt', 'r') as f:
    text_biomech = f.read()
abstracts_biomech = text_biomech.split('\n\n')
abstracts_biomech = [a.strip() for a in abstracts_biomech if a.strip()]


print("Number of Biochemistry abstracts:", len(abstracts_biochem))
print("Number of Biomechanics abstracts:", len(abstracts_biomech))

embeddings_biochem = []
embeddings_biomech = []

# Generate embeddings for Biochemistry abstracts
for i in tqdm(range(0, len(abstracts_biochem), 256)):
    embeddings_biochem.extend(model.encode(abstracts_biochem[i:i+256]))

# Save the embeddings to a file
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/biobertLEMembeddings_biochem.pkl', 'wb') as f:
    pickle.dump(embeddings_biochem, f)

for i in tqdm(range(0, len(abstracts_biomech), 256)):
    embeddings_biomech.extend(model.encode(abstracts_biomech[i:i+256]))

with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/biobertLEMembeddings_biomech.pkl', 'wb') as f:
    pickle.dump(embeddings_biomech, f)
