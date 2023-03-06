import os
import re
import spacy
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# Function to process abstracts
def process_abstracts(abstracts_chunk):
    nlp = spacy.load("en_core_sci_sm")
    proteins_chunk = set()

    for abstract in abstracts_chunk:
        doc = nlp(abstract)

        for entity in doc.ents:
            if entity.label_ in ['PROTEIN', 'GENE_OR_GENE_PRODUCT']:
                proteins_chunk.add(entity.text)

    return proteins_chunk


# Set the path to the folder containing the text files
folder_path = '/path/to/folder/'

# Create a list to store the abstracts
abstracts = []

# Iterate over all files in the folder
for filename in tqdm(os.listdir(folder_path)):
    if filename.endswith('.txt'):
        with open(os.path.join(folder_path, filename), 'r') as f:
            text = f.read()
        abstracts.extend(re.split(r'\n\s*\n', text))

# Determine the number of CPU cores
num_cores = cpu_count()

# Calculate the number of abstracts to process per core
chunk_size = len(abstracts) // num_cores

# Split the abstracts into chunks
abstracts_chunks = [abstracts[i:i + chunk_size] for i in range(0, len(abstracts), chunk_size)]

# Create a multiprocessing pool and process the abstracts
with Pool(num_cores) as pool:
    proteins_list = pool.map(process_abstracts, abstracts_chunks)

# Combine the proteins sets from all cores into a single set
proteins = set.union(*proteins_list)

# Write the protein names to a text file
with open('proteins.txt', 'w') as f:
    for protein in tqdm(proteins):
        f.write(protein + '\n')
