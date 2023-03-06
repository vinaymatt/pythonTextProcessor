import os

input_file = "/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/PII_Elsevier/output_file.txt"
output_directory = "/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/PII_Elsevier"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Read the elements from the input file
with open(input_file, "r") as file:
    elements = [line.strip() for line in file.readlines()]

# Function to divide the list into chunks of 4,000 elements
def divide_chunks(l, n=4000):
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Divide the elements into chunks of 4,000
chunks = list(divide_chunks(elements))

# Write each chunk to a separate file in the specified output directory
for index, chunk in enumerate(chunks, start=1):
    output_file = os.path.join(output_directory, f"output_{index}.txt")
    with open(output_file, "w") as file:
        for item in chunk:
            file.write(f"{item}\n")
