import spacy

def process_abstracts(abstracts_chunk):
    nlp = spacy.load("en_core_sci_sm")
    proteins_chunk = set()

    for abstract in abstracts_chunk:
        doc = nlp(abstract)

        for entity in doc.ents:
            if entity.label_ in ['PROTEIN', 'GENE_OR_GENE_PRODUCT']:
                proteins_chunk.add(entity.text)

    return proteins_chunk
