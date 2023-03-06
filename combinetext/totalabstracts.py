import os

folder = "/Users/vinay/PycharmProjects/pythonTextProcessor/text_processor/Actual All Abstracts"  # Replace with your folder path

# Get all the file names in the folder
file_names = os.listdir(folder)

# Create a dictionary to store the abstracts by year
abstracts_by_year = {}

# Iterate over the file names and extract abstracts by year
for file_name in file_names:
    if not file_name.endswith(".txt"):
        continue

    year = file_name[:4]
    file_path = os.path.join(folder, file_name)

    with open(file_path, "r") as f:
        lines = f.readlines()
        abstracts = [line.strip() for line in lines if line.strip()]

    if year not in abstracts_by_year:
        abstracts_by_year[year] = []

    abstracts_by_year[year].append((file_name, abstracts))

# Iterate over the abstracts by year and compare abstracts between files
total_abstracts = 0
counter = 0
abstracts_at_64 = 0
abstracts_after_64 = 0

for year, file_abstracts in abstracts_by_year.items():
    print(f"Year {year}:")
    for file_name, abstracts in file_abstracts:
        num_abstracts = len(abstracts)
        total_abstracts += num_abstracts
        print(f"- File {file_name} has {num_abstracts} abstracts.")

        if counter == 55:
            abstracts_at_64 = total_abstracts

        if counter > 55:
            abstracts_after_64 += num_abstracts

        counter += 1
    print()

print(f"Total number of abstracts: {total_abstracts}")
print(f"Total number of abstracts in the 64th index: {abstracts_at_64}")
print(f"Total number of abstracts after the 64th index: {abstracts_after_64}")
