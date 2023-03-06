import requests
from lxml import etree
from tqdm import tqdm

def fetch_abstract(pmid):
    URL = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id={pmid}'
    response = requests.get(URL)
    root = etree.fromstring(response.content)
    abstract = root.xpath("//AbstractText")
    return abstract[0].text if len(abstract) > 0 else ""

def fetch_all_abstracts(term):
    URL = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=100000&term={term}'
    response = requests.get(URL)
    pmids = response.json()['esearchresult']['idlist']

    all_abstracts = []
    for pmid in tqdm(pmids):
        abstract = fetch_abstract(pmid)
        all_abstracts.append(abstract)

    return all_abstracts

abstracts = fetch_all_abstracts('Tissue Engineering AND 2014/01/01:2014/12/31[dp]')
with open("2014_Abstracts.txt", "w", encoding='UTF-8') as f:
    f.write('\n'.join(abstracts))
