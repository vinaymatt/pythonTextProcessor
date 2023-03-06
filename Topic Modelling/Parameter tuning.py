import os
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import HdpModel

def load_saved_data(output_directory, corpus_file, dictionary_file):
    corpus = MmCorpus(os.path.join(output_directory, corpus_file))
    dictionary = Dictionary.load(os.path.join(output_directory, dictionary_file))
    return corpus, dictionary

def train_hdp(corpus, dictionary):
    hdp_model = HdpModel(corpus, dictionary)
    return hdp_model

def main():
    output_directory = "/Users/vinay/PycharmProjects/pythonTextProcessor/data"

    # Load saved data
    biomech_corpus, biomech_dictionary = load_saved_data(output_directory, "biomech_corpus.mm", "biomech_dictionary.dict")
    biochem_corpus, biochem_dictionary = load_saved_data(output_directory, "biochem_corpus.mm", "biochem_dictionary.dict")

    # Train HDP models
    biomech_hdp_model = train_hdp(biomech_corpus, biomech_dictionary)
    biochem_hdp_model = train_hdp(biochem_corpus, biochem_dictionary)

    # Save the trained HDP models
    biomech_hdp_model.save(os.path.join(output_directory, "biomech_hdp_model.hdp"))
    biochem_hdp_model.save(os.path.join(output_directory, "biochem_hdp_model.hdp"))

if __name__ == "__main__":
    main()
