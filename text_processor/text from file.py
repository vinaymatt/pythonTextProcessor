import re

abstract_count = 0

input_file = '/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge/combined_file.txt'
output_file = '/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge/output.txt'

# Initialize an empty set to keep track of extracted abstracts
abstracts = set()

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        if line.startswith('Abstract:'):
            # Extract the abstract
            abstract = re.sub('^Abstract:\s*', '', line.strip())
            # Check if the abstract has already been extracted
            if abstract not in abstracts:
                # Write the abstract to the output file
                f_out.write(abstract + '\n\n')
                # Add the abstract to the set
                abstracts.add(abstract)
                # Increment the abstract count
                abstract_count += 1

print(f'Extracted {abstract_count} abstracts.')




