import os
import re

directory = "/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge"

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filepath = os.path.join(directory, filename)
        with open(filepath, "r") as f:
            text = f.read()

        text = re.sub(r'\b(Objective:|Background:|Introduction:)\b', '', text)
        text = re.sub(r'\bElsevier Ltd[a-zA-Z]+\b', '', text)
        text = re.sub(r'\bCopyright[a-zA-Z\s]+Elsevier\b', '', text)
        text = re.sub(r'\bElsevier Inc[a-zA-Z\s]+\b', '', text)
        text = re.sub(r'\bElsevier [a-zA-Z\s]+\b', '', text)
        text = re.sub(r"© \d{4}", "", text)
        text = re.sub(r"The Author(s)[a-zA-Z\s] \d{4}", "", text)
        text = re.sub(r"The Authors[a-zA-Z\s] \d{4}", "", text)

        with open(filepath, "w") as f:
            f.write(text)