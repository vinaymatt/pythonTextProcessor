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

    abstracts_by_year[year].extend(abstracts)

# Write the unique abstracts to files for each year
for year, abstracts in abstracts_by_year.items():
    unique_abstracts = list(set(abstracts))
    output_file = os.path.join(folder, f"{year}_abstracts.txt")
    with open(output_file, "w") as f:
        for abstract in unique_abstracts:
            f.write(f"{abstract}\n\n")

    # Delete all files for the year except the output file
    for file_name in file_names:
        if file_name.startswith(year) and file_name != f"{year}_abstracts.txt":
            os.remove(os.path.join(folder, file_name))
