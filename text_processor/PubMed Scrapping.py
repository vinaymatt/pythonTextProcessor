import os
from Bio import Entrez
from tqdm import tqdm

# Replace with your email and API key
Entrez.email = "vvm5242@psu.edu"
Entrez.api_key = "27840c1f80d4b0de46f0e3c97e40e98edc09"

# Set the range of years you want to search
start_year = 1984
end_year = 2022

# Query term for biomechanics
query_term = "biomechanics"

for year in range(start_year, end_year + 1):
    year_query = f"{query_term} AND {year}[pdat]"
    search_results = Entrez.read(Entrez.esearch(db="pubmed", term=year_query, retmax=100000))
    total_results = int(search_results["Count"])
    print(f"Searching for articles in {year}...")
    print(total_results)
    with tqdm(total=total_results) as pbar:
        dois = []

        for start in range(0, total_results, 100):
            search_results = Entrez.read(Entrez.esearch(db="pubmed", term=year_query, retstart=start, retmax=100))
            id_list = search_results["IdList"]

            for id in id_list:
                summary = Entrez.read(Entrez.esummary(db="pubmed", id=id))
                doi = summary[0].get("DOI", "")

                if doi:
                    dois.append(doi)

                pbar.update(1)

    # Save DOIs to a text file named by the year
    with open(f"{year}_DOIs.txt", "w") as f:
        for doi in dois:
            f.write(f"{doi}\n\n")
