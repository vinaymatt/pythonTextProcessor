from Bio import Medline
import os

def process_nbib_files(nbib_dir, output_dir):
    # Loop through all nbib files in the directory
    for nbib_file in os.listdir(nbib_dir):
        if nbib_file.endswith(".nbib"):
            nbib_path = os.path.join(nbib_dir, nbib_file)
            # Read the NBIB file
            with open(nbib_path, "r") as f:
                records = Medline.parse(f)

                # Extract abstracts and store them by year
                abstracts_by_year = {}
                for record in records:
                    year = record.get("DP", "").split()[0]  # Extract the publication year
                    abstract = record.get("AB", "")

                    if year and abstract:
                        if year not in abstracts_by_year:
                            abstracts_by_year[year] = []

                        abstracts_by_year[year].append(abstract)

            # Create a folder for the output files if it doesn't exist
            output_folder = os.path.join(output_dir, nbib_file.replace(".nbib", ""))
            os.makedirs(output_folder, exist_ok=True)

            # Save abstracts in separate files based on the year
            for year, abstracts in abstracts_by_year.items():
                output_file = os.path.join(output_folder, f"{year}_abstracts.txt")
                with open(output_file, "w") as f:
                    for abstract in abstracts:
                        f.write(abstract)
                        f.write("\n\n")


nbib_dir = "/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/Abstracts_DrButler/untitled folder 5"
output_dir = "/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/Abstracts_DrButler/untitled folder 5/Abstracts"

process_nbib_files(nbib_dir, output_dir)
