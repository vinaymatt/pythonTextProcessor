import os
import filecmp

# Path of the directory to search
directory_path = '/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge'

# Get a list of all the text files in the directory
text_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]

count = 0

# Compare each pair of text files in the directory
for i in range(len(text_files)):
    for j in range(i + 1, len(text_files)):
        file1 = os.path.join(directory_path, text_files[i])
        file2 = os.path.join(directory_path, text_files[j])

        # Check if the files exist before comparing them
        if os.path.exists(file1) and os.path.exists(file2):
            # Use filecmp to compare the contents of the two files
            if filecmp.cmp(file1, file2):
                # Determine which file to delete based on the file name
                if int(text_files[i][:-4]) > int(text_files[j][:-4]):
                    delete_file = os.path.join(directory_path, text_files[i])
                else:
                    delete_file = os.path.join(directory_path, text_files[j])

                # Delete the higher-valued file
                os.remove(delete_file)
                count += 1

print(f"Deleted {count} duplicate files.")
