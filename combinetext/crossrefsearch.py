import requests
import time
import json
from pathlib import Path
from tqdm import tqdm

api_key = "175fc0d14b1c41fa38518209fbd360f1"
query = "tissue engineering"
base_url = f"http://api.springernature.com/meta/v2/json?q={query}&p=100&api_key={api_key}"
output_directory = "/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/Abstracts_DrButler/springer/Tissue engineering"

# Define relevant keywords for biomechanics
keywords = [  "biomechanical",  "biomechanics",    "mechanics",    "motion",    "movement",    "kinematics",    "dynamics",    "forces",    "stress",    "strain",    "musculoskeletal",    "biomechanical",    "tissue mechanics",    "gait",    "ergonomics",    "posture",    "locomotion",    "kinesiology",    "muscle",    "tendon",    "ligament",    "bone",    "joint",    "orthopedics",    "rehabilitation",    "prosthetics",    "orthotics",    "biomaterials",    "implants",    "sports biomechanics",    "human factors",    "computational biomechanics",    "finite element",    "multibody dynamics",    "biomedical engineering",    "biofluid mechanics",    "cardiovascular biomechanics",    "pulmonary biomechanics",    "cellular biomechanics",    "biotribology"]

def get_total_results():
    url = f"{base_url}&s=1"
    response = requests.get(url)
    response.raise_for_status()
    result = response.json()
    return int(result['result'][0]['total'])


def is_biomechanics_related(abstract):
    abstract_lower = abstract.lower()
    return True


def save_abstract(abstract, year):
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    file_name = f"{year}_abstracts.txt"
    file_path = output_path / file_name

    with open(file_path, "a") as file:
        file.write(abstract + "\n")


def fetch_and_save_abstracts(offset, total_results):
    saved_abstracts = 0
    url = f"{base_url}&s={offset}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error while fetching search results: {e}")
        return saved_abstracts

    for record in result["records"]:
        if "publicationDate" in record and "abstract" in record:
            year = record["publicationDate"][:4]
            abstract = record["abstract"]
            if is_biomechanics_related(abstract):
                save_abstract(abstract, year)
                saved_abstracts += 1

    return saved_abstracts


def main(resume_offset=None):
    total_results = get_total_results()
    total_hits = 0
    total_abstracts_saved = 0

    print(f"Total hits during the search: {total_results}")

    if not resume_offset:
        resume_offset = 1

    progress_bar = tqdm(total=total_results, initial=int(max(0, resume_offset - 1)), desc="Fetching results")

    while resume_offset <= total_results:
        saved_abstracts = fetch_and_save_abstracts(resume_offset, total_results)
        total_abstracts_saved += saved_abstracts
        total_hits += 100
        resume_offset += 100
        progress_bar.update(100)
        time.sleep(1)

    progress_bar.close()

    print(f"Total abstracts saved: {total_abstracts_saved}")


if __name__ == "__main__":
    main(resume_offset=266601)  # Pass the offset to resume from, if needed.
