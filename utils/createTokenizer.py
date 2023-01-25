from spacy.lang.en import English
import spacy
from owlready2 import *
import gensim
import json
from tqdm import tqdm
import spacy
import logging

logging.basicConfig(filename="Test.txt", filemode="w", level=logging.INFO)

from spacy.attrs import ORTH



def createTokenizer(dir, total, nlp):
    new_data = []
    nlp.tokenizer.from_disk("./tokenizer/tokenizer")
    with open(dir) as f:
        data = json.load(f)
    for idx in tqdm(range(total)):
        text = data[idx]["text"]
        label = data[idx]["relation"]
        gene_pos = data[idx]["h"]["pos"]
        dis_pos = data[idx]["t"]["pos"]
        orgintextVSann(text, gene_pos, nlp, idx)
        orgintextVSann(text, dis_pos, nlp, idx)
    f.close()
    nlp.tokenizer.to_disk("./tokenizer/tokenizer")



def orgintextVSann(textstr, pos, nlp, i):
    ent = textstr[pos[0]:pos[0] + pos[1]]
    #if " " in ent: return None
    words = textstr.split(" ")
    pointer = 0
    idx = 0

    entFound = ""
    new_pos=[0, len(words[0])]
    while pointer <= pos[0]:


        new_pos[0] = pointer
        new_pos[1] = len(words[idx])
        pointer += len(words[idx]) + 1  # Add the length of current substring and one blank space



        if pointer > pos[0]:
            entFound = words[idx]

        idx += 1

    # print(textstr)
    # print(entFound)
    # print(textstr[new_pos[0]: new_pos[0] + new_pos[1]])
    # print(ent)
    # print(new_pos)
    # print(pos)
    assert entFound == textstr[new_pos[0]: new_pos[0] + new_pos[1]]
    # if entFound.endswith(",") or entFound.endswith("."):
    #     return None
    if entFound != ent:                                       # and (entFound not in ent)
        try:
            if new_pos[0] == pos[0]:                          # CD4(+)   CD4
                secHalf = entFound[len(ent):]
                case = [{ORTH: ent}, {ORTH: secHalf},]
                nlp.tokenizer.add_special_case(entFound, case)
                logging.info("First senario: we found {} in text, {} is the entity, and second half are {}".format(entFound, ent, secHalf))
            elif sum(new_pos) == sum(pos):
                firstHalf = entFound[:-len(ent)]
                case = [{ORTH: firstHalf}, {ORTH:ent}, ]
                nlp.tokenizer.add_special_case(entFound, case)
                logging.info(
                    "Second senario: we found {} in text, {} is the entity, and first half are {}".format(entFound, ent,
                                                                                                          firstHalf))
            elif new_pos[0] < pos[0] and (sum(new_pos)) > (sum(pos)):

                first = entFound[:pos[0] - new_pos[0]]
                last = entFound[sum(pos) - new_pos[0]:]
                case = [{ORTH: first}, {ORTH: ent}, {ORTH: last}]
                nlp.tokenizer.add_special_case(entFound, case)
                #
                logging.info(
                    "Third senario: we found {} in text, {} is the entity, and first half are {}, last part are {}".format(entFound, ent,
                                                                                                          first, last))
        except Exception as e:
            print(e)
            print("*****************************************")
            print(textstr)
            print("In the text: {}".format(entFound))
            print("In Annotation: {}".format(ent))
            print("Line: {}".format(i))
            print("*****************************************\n")



    return None


if __name__ == '__main__':
    nlp = English()
    case = [{ORTH: "cytokine-"}, {ORTH: "cytokine"}]
    nlp.tokenizer.add_special_case("cytokine-cytokine", case)
    case = [{ORTH: "3-kinase/"}, {ORTH: "protein"}]
    nlp.tokenizer.add_special_case("3-kinase/protein", case)
    case = [{ORTH: "(NG2(+)"}, {ORTH: "cytokine"}]
    nlp.tokenizer.add_special_case("(NG2(+)cytokine", case)
    case = [{ORTH: "TEL/"}, {ORTH: "ART"}]
    nlp.tokenizer.add_special_case("TEL/ART", case)
    case = [{ORTH: "PI3K/"}, {ORTH: "AKT"},{ORTH: "/mTOR"}]
    nlp.tokenizer.add_special_case("PI3K/AKT/mTOR", case)
    case = [{ORTH: "PICALM-"},{ORTH: "MLLT10"}]
    nlp.tokenizer.add_special_case("PICALM-MLLT10", case)
    case = [{ORTH: "CD26/"},{ORTH: "DPPIV"}]
    nlp.tokenizer.add_special_case("CD26/DPPIV", case)
    case = [{ORTH: "Delta"},{ORTH: "FosB"}]
    nlp.tokenizer.add_special_case("DeltaFosB", case)
    nlp.tokenizer.to_disk("./tokenizer/tokenizer")


    createTokenizer("data/TBGA/TBGA_train_processed_subset.json", 37249, nlp)
    createTokenizer("data/TBGA/TBGA_val_processed.json", 6587, nlp)
    createTokenizer("data/TBGA/TBGA_test_processed.json", 6547, nlp)
    #example = "Although the PI3K/AKT/mTOR signaling pathway plays a role in MB, we did not find TSC1/TSC2 (TSC, tuberous sclerosis complex) mutation in our patient."
    #orgintextVSann(example, [18, 3], nlp, 0)
    # example = "Dietary intake of phytoestrogens, estrogen receptor-beta polymorphisms and the risk of prostate cancer."
    # print(example[34:56])

















    # Self Test
    # example = "To observe the dynamic expression of GDNF and their receptors in the brain of rats after status epilepticus(SE)."
    # words = example.split(" ")
    # pointer = 0
    # idx = 0
    # ent = example[89:107]
    # print(ent)
    # entFound = ""
    # new_pos_s = 0
    # new_pos_t = len(words[0])
    # print(example)
    # while pointer <= 89:
    #
    #     print("Idx {}---------------------------------".format(idx))
    #     print("Current word: {}".format(words[idx]))
    #     new_pos_s = pointer
    #     new_pos_t = len(words[idx])
    #     print("Word position {} - {}".format(new_pos_s, new_pos_t))
    #     pointer += new_pos_t + 1  # Add the length of current substring and one blank space
    #     print("New pointer {}".format(pointer))
    #
    #     if pointer > 89:
    #         entFound = words[idx]
    #         print(entFound)
    #
    #     idx += 1

