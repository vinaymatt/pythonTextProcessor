import requests
import re
import os

def get_abstract(doi, api_keys):
    for api_key in api_keys:
        url = f"https://api.elsevier.com/content/abstract/doi/{doi}"
        headers = {
            "Accept": "application/json",
            "X-ELS-APIKey": api_key
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            abstract = response.json().get("abstracts-retrieval-response").get("coredata").get("dc:description")
            return abstract
        else:
            continue
    return None

api_keys = ["9d48ca0e7edbe5ce0b31eb17d90b982c", "aca7f7693030f5da4613c31ac78033f1", "853dae6b071772a96ce04dd548734a8e", "9279a4699c3ee5af84834cb62d6f5e46", "067cb8dab3c5aef0665b5e4f64eab48b", "3a98635601173a3b6f07e543989299fc", "be19de75a260312dea273283c179e7eb", "eec4806bfae52c6f6885b4f44b9d4a82", "d9774184edc593d1ea5eef2fc25be250"]
directory = '/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge'

with open('/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge/combined_file.txt', 'r') as file:
    text = file.read()

doi = set(re.findall(r'https://doi.org/(10\.\d{4}/.+)\b', text))
doi = list(doi)
print(doi)
print(len(doi))

for index, doi in enumerate(doi):
    abstract = get_abstract(doi, api_keys)
    if abstract:
        filename = "{:05d}.txt".format(index + 1)
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            with open(filepath, "w") as f:
                f.write(abstract + "\n")
    else:
        print(f"Could not find abstract for {doi}")
        continue
