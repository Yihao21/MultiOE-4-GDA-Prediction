from spacy.lang.en import English
import spacy
from owlready2 import *
import gensim


def create_tagger(onto, onto_type, pat_li):
    '''

    :param onto:   ontology loaded by owlready2
    :param label:  str Gene|disease type of ontology
    :param pat_li empty list for patterns
    :return:
    '''
    for c in onto.classes():
        label = str(c.label)
        if label == "[]":
            continue
        else:
            label = label.replace("['","").replace("']","")
            subwords = label.split(" ")

            pattern = {"label": onto_type, "pattern":[{"LOWER": subwords[0].lower()}], "id": c.iri}
            if len(subwords) > 1:
                for idx in range (1, len(subwords)):
                    pattern["pattern"].append({"LOWER":subwords[idx].lower()})
            pat_li.append(pattern)
    return pat_li

def print_listofdict(li):
    for di in li:
        for key, value in di.items():
            print(key)
            print(value)

def main():
    nlp = English()
    ruler = nlp.add_pipe("entity_ruler")
    nlp = spacy.blank('en')  # create blank Language class

    onto_doid = get_ontology("onto/doid-base.owl").load()
    patterns = []
    doid_patterns = create_tagger(onto_doid, "DISEASE", pat_li=patterns)

    onto_ogg = get_ontology("onto/ogg.owl").load()
    all_patterns = create_tagger(onto_ogg, "GENE", pat_li=patterns)

    ruler.add_patterns(all_patterns)
    ruler.to_disk("ruler/ogg_doid")

    model = gensim.models.Word2Vec.load("E:\Fraunhofer SCAI\SCAI\OWL2Vec-Star-master\cache\oggdoid\output.embeddings")
    print(model.wv.get_vector("http://purl.obolibrary.org/obo/OGG_3000008644"))  #http://purl.obolibrary.org/obo/GO_0019807




if __name__ == '__main__':
    main()

