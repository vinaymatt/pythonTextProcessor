import os
import re
from langdetect import detect

folder_path = '/Users/vinay/PycharmProjects/pythonTextProcessor/text_processor/Actual All Abstracts'
output_folder_path = '/Users/vinay/PycharmProjects/pythonTextProcessor/text_processor/English_Abstracts'

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        with open(os.path.join(folder_path, filename), 'r') as f:
            text = f.read()

        abstracts = re.split(r'\n\s*\n', text)
        english_abstracts = []

        for abstract in abstracts:
            try:
                language = detect(abstract)
            except:
                language = 'unknown'

            if language == 'en':
                english_abstracts.append(abstract)

        if english_abstracts:
            with open(os.path.join(output_folder_path, f"english_{filename}"), 'w') as f:
                f.write('\n\n'.join(english_abstracts))
