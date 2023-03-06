import os
import re
import spacy
from tqdm import tqdm

# Load the SciSpacy models
nlp1 = spacy.load('en_ner_jnlpba_md')
nlp2 = spacy.load('en_ner_bionlp13cg_md')

# Set the path to the folder containing the text files
folder_path = '/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/Temp'

# Create dictionaries to store the extracted entities
proteins = set()
genes = set()
cell_types = set()
anatomical_structures = set()
cellular_components = set()

# Initialize tqdm with the total number of files
total_files = sum(1 for filename in os.listdir(folder_path) if filename.endswith('.txt'))
pbar_files = tqdm(total=total_files, desc='Files processed')

# Set the starting file number
start_file_number = 0

# Iterate over all files in the folder
for idx, filename in enumerate(sorted(os.listdir(folder_path))):
    # Check if the file is a text file
    if filename.endswith('.txt') and idx >= start_file_number:
        # Update the tqdm progress bar
        pbar_files.update(1)

        # Read in the text file as a string
        with open(os.path.join(folder_path, filename), 'r') as f:
            text = f.read()

        # Split the text into individual abstracts
        abstracts = re.split(r'\n\s*\n', text)

        # Iterate over each abstract
        for abstract in abstracts:
            # Process the abstract using the first Spacy model
            doc1 = nlp1(abstract)

            # Extract named entities using the first NER model
            for ent in doc1.ents:
                if ent.label_ == 'PROTEIN':
                    proteins.add(ent.text)
                elif ent.label_ == 'CELL_TYPE':
                    cell_types.add(ent.text)

            # Process the abstract using the second Spacy model
            doc2 = nlp2(abstract)

            # Extract named entities using the second NER model
            for ent in doc2.ents:
                if ent.label_ == 'GENE_OR_GENE_PRODUCT':
                    genes.add(ent.text)
                elif ent.label_ == 'ANATOMICAL_SYSTEM' or ent.label_ == 'CELLULAR_COMPONENT':
                    anatomical_structures.add(ent.text)
                elif ent.label_ == 'ORGANISM_SUBSTANCE':
                    cellular_components.add(ent.text)

            # Save the extracted entities to separate files
            with open('/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/proteins.txt', 'w') as f:
                for entity in proteins:
                    f.write(entity + '\n')

            with open('/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/genes.txt', 'w') as f:
                for entity in genes:
                    f.write(entity + '\n')

            with open('/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/cell_types.txt', 'w') as f:
                for entity in cell_types:
                    f.write(entity + '\n')

            with open('/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/anatomical_structures.txt', 'w') as f:
                for entity in anatomical_structures:
                    f.write(entity + '\n')

            with open('/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/cellular_components.txt', 'w') as f:
                for entity in cellular_components:
                    f.write(entity + '\n')

# Close the tqdm progress bar
pbar_files.close()


