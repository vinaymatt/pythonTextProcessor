import requests
import json
import time
from tqdm import tqdm

# Load the API keys from the file
with open("/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/PII_Elsevier/piis.txt", "r") as api_keys_file:
    API_KEYS = [key.strip() for key in api_keys_file.readlines()]

# Load the PIIs from the file
with open("/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/PII_Elsevier/output_file.txt", "r") as piis_file:
    PIIs = [pii.strip() for pii in piis_file.readlines()]

def get_abstract(pii, api_key):
    url = f"https://api.elsevier.com/content/article/pii/{pii}"
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json",
    }
    params = {"view": "META_ABS"}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        try:
            content = json.loads(response.text)
            abstract = content["full-text-retrieval-response"]["coredata"]["dc:description"]
            year = content["full-text-retrieval-response"]["coredata"]["prism:coverDate"][:4]
            return abstract, year
        except (KeyError, json.JSONDecodeError):
            return None, None
    else:
        return None, None

def save_abstract(abstract, year):
    with open(f"{year}_abstracts.txt", "a") as abstracts_file:
        abstracts_file.write(abstract + "\n")

def download_abstracts(start=0):
    api_key_index = 0
    for i in tqdm(range(start, len(PIIs)), desc="Downloading abstracts"):
        pii = PIIs[i]
        abstract, year = get_abstract(pii, API_KEYS[api_key_index])

        while abstract is None and api_key_index < len(API_KEYS) - 1:
            api_key_index += 1
            abstract, year = get_abstract(pii, API_KEYS[api_key_index])

        if abstract is not None:
            save_abstract(abstract, year)
        else:
            continue

        api_key_index = 0
        time.sleep(1)

# To resume the process, update the 'start' parameter with the index of the PII where the process stopped last.
# For example, if the process stopped at the 10th PII, set start=10.
download_abstracts(start=1)
