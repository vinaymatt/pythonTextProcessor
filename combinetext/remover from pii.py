import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Constants
API_KEY = '853dae6b071772a96ce04dd548734a8e'
FILE_NAME = '/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/PII_Elsevier/output_file.txt'
OUTPUT_FILE = '/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/PII_Elsevier/valpiis.txt'
ELSEVIER_URL = 'https://api.elsevier.com/content/article/pii/'
HEADERS = {'X-ELS-APIKey': API_KEY}
MAX_WORKERS = 10


# Function to check if a PII exists
def pii_exists(pii):
    url = f'{ELSEVIER_URL}{pii}'
    response = requests.get(url, headers=HEADERS)
    return response.status_code == 200


# Read PIIs from the input file
with open(FILE_NAME, 'r') as file:
    piis = [line.strip() for line in file.readlines()]

# Check if PIIs exist in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_pii = {executor.submit(pii_exists, pii): pii for pii in piis}
    valid_piis = []

    for future in tqdm(as_completed(future_to_pii), total=len(piis), desc="Checking PIIs"):
        pii = future_to_pii[future]
        try:
            exists = future.result()
            if exists:
                valid_piis.append(pii)
        except Exception as exc:
            print(f'An error occurred while checking PII {pii}: {exc}')

# Write valid PIIs to the output file
with open(OUTPUT_FILE, 'w') as file:
    for pii in valid_piis:
        file.write(f'{pii}\n')

print(f"Valid PIIs have been saved in '{OUTPUT_FILE}'.")
