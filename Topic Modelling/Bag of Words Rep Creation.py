import os
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel
from gensim.matutils import corpus2csc

def read_abstracts(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        abstracts = content.split('\n\n')
    return abstracts

def create_bow_representation(abstracts):
    tokenized_abstracts = [abstract.split() for abstract in abstracts]
    dictionary = Dictionary(tokenized_abstracts)
    dictionary.filter_extremes(no_below=5)
    corpus = [dictionary.doc2bow(abstract) for abstract in tokenized_abstracts]
    return corpus, dictionary

def main():
    biomech_file_path = "/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biomechlarge/preprocessedword2vec_withTFID_abstracts_biomech.txt"
    biochem_file_path = "/Users/vinay/PycharmProjects/pythonTextProcessor/data/Biochemlarge/preprocessedword2vec_withTFID_abstracts_biochem.txt"

    output_directory = "/Users/vinay/PycharmProjects/pythonTextProcessor/data"

    # Read preprocessed abstracts
    biomech_abstracts = read_abstracts(biomech_file_path)
    biochem_abstracts = read_abstracts(biochem_file_path)

    # Create Bag-of-Words representation for both sets of abstracts
    biomech_corpus, biomech_dictionary = create_bow_representation(biomech_abstracts)
    biochem_corpus, biochem_dictionary = create_bow_representation(biochem_abstracts)

    # Save the results
    biomech_dictionary.save(os.path.join(output_directory, "biomech_dictionary.dict"))
    biochem_dictionary.save(os.path.join(output_directory, "biochem_dictionary.dict"))

    MmCorpus.serialize(os.path.join(output_directory, "biomech_corpus.mm"), biomech_corpus)
    MmCorpus.serialize(os.path.join(output_directory, "biochem_corpus.mm"), biochem_corpus)

if __name__ == "__main__":
    main()

