import os

def combine_files(folder_path, output_file):
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r') as infile:
                    outfile.write(infile.read() + "\n")

folder_path = "/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge"
output_file = "/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge/AllAbstracts.txt"
combine_files(folder_path, output_file)