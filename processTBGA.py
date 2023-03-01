from owlready2 import *
import json
from tqdm import tqdm
import pandas as pd
from spacy.lang.en import English
from spacy.training import offsets_to_biluo_tags
import warnings
import random

random.seed(29)

def process_raw_txt(dir, targetdir):
    data_li = []
    with open(dir,"r") as f:
        for line in f:
            j_content = json.loads(line)
            data_li.append(j_content)
    f.close()
    print_raw(data_li, targetdir)


def find_name(onto, labelname):
    for idx, c in enumerate(onto):
        tmp = str(c.label).replace("['","").replace("']","")
        if tmp == labelname:
            return c.iri
        else:
            continue
    return labelname

def process_jsondata_gene(dir, onto, total):
    non_idx = []
    with open(dir) as f:
        data = json.load(f)
    f.close()
    for idx in tqdm(range(total)):
        gene_name = data[idx]["h"]["name"]
        iri = find_name(onto, gene_name)
        if iri == gene_name:
            non_idx.append(idx)
        else:
            data[idx]["h"]["id"] = iri

    with open("data/TBGA/filenew.json", "w") as fw:
       json.dump(data, fw)

    fw.close()
    print(non_idx)

def find_umls(dis_id, df):
    for idx, umls in enumerate(df.mapped_curie):
        if dis_id in umls:
            s = str(df.curie_id[idx]).replace(":","_")
            irilabel = "http://purl.obolibrary.org/obo/" + s
            return irilabel

    return "not found"

def process_jsondata_dis(dir, dataframe, total):
    non_idx = []
    with open(dir) as f:
        data = json.load(f)
    for idx in tqdm(range(total)):
        dis_id = data[idx]["t"]["id"]
        irilabel = find_umls(dis_id, dataframe)
        if irilabel == "not found":
            non_idx.append(idx)
        else:
            data[idx]["t"]["id"] = irilabel
    f.close()

    with open("data/TBGA/filenew.json", "w") as fw:
       json.dump(data, fw)

    fw.close()
    print(non_idx)

def re_print(dir, targetdir):
    with open(dir) as f:
        data = json.load(f)
    print_raw(data, targetdir)
    f.close()

def print_raw(data, targetdir):
    with open(targetdir, 'w') as fp:
        fp.write(
            '[' +
            ',\n'.join(json.dumps(i) for i in data) +
            ']\n')
    fp.close()

def extractProcessedId(dir, targetdir, total):
    new_data = []
    with open(dir) as f:
        data = json.load(f)
    for idx in tqdm(range(total)):
        gene_id = str(data[idx]["h"]["id"])
        dis_id = str(data[idx]["t"]["id"])
        if gene_id.startswith("http") and dis_id.startswith("http"):
            new_data.append(data[idx])
    f.close()

    with open(targetdir, "w") as fw:
        json.dump(new_data, fw)

    fw.close()

#Create subset of the dataset depends on specific class
def createSubset(dir, l, targetdir, total):
    new_data=[]
    with open(dir) as f:
        data = json.load(f)
    for idx in tqdm(range(total)):
        label = data[idx]["relation"]
        # gene_pos = data[idx]["h"]["pos"]
        # gene_id = data[idx]["h"]["id"]
        # dis_pos = data[idx]["t"]["pos"]
        # dis_id = data[idx]["t"]["id"]

        if label == l:
            number = random.randint(0, 100)
            if number <= 80:
                new_data.append(data[idx])

        else:
            new_data.append(data[idx])

    print(len(new_data))
    with open(targetdir, "w") as fw:
        json.dump(new_data, fw)

    fw.close()

def replacespecial(dir, targetdir, total):
    new_data = []
    with open(dir) as f:
        data = json.load(f)
    for idx in tqdm(range(total)):
        text = data[idx]["text"]
        new_t = text.replace("("," ").replace(")"," ")
        new_data.append(data[idx])
        new_data[idx]["text"] = new_t
    f.close()

    with open(targetdir, "w") as fw:
        json.dump(new_data, fw)

    fw.close()

def check_sentence_length(dir):
    with open(dir) as f:
        data = json.load(f)
    total = len(data)
    #ok_length = sum(len(str(samples["text"])) < 64 for samples in data)
    #print("The number of samples with length smaller than 64 is: {}/{}".format(ok_length, total))
    subdata = [samples for samples in data if len(str(samples["text"])) <= 128]
    with open("data/TBGA/TBGA_shorttextset.json", "w") as fw:
        json.dump(subdata, fw)

    fw.close()

def avg_len(dir):
    with open(dir) as f:
        data = json.load(f)
    total = len(data)
    subdata = [len(str(samples["text"])) for samples in data]
    print(sum(subdata) / total)

def countclasses(dir):
    with open(dir) as f:
        data = json.load(f)
    total = len(data)
    count_class = {"NA": 0, "therapeutic": 0, "biomarker": 0, "genomic_alterations": 0}
    for idx in tqdm(range(total)):
        l = data[idx]["relation"]
        if l == "NA":
            count_class["NA"] += 1
        elif l == "therapeutic":
            count_class["therapeutic"] += 1
        elif l == "biomarker":
            count_class["biomarker"] += 1
        elif l == "genomic_alterations":
            count_class["genomic_alterations"] += 1
    for k in count_class.keys():
        print('{} has {} samples. {:.4f}'.format(k, str(count_class[k]), count_class[k] / total))

    print('Number of samples: {}'.format(total))

def main():
    process_raw_txt("data/TBGA/TBGA_test.json", "data/TBGA/test1.json")
    print("Processing raw file")
    ogg = get_ontology("onto/ogg.owl").load()
    classes_ogg = list(ogg.classes())
    doid = get_ontology("onto/doid-base.owl").load()
    classes_doid = list(doid.classes())
    print("ontology loaded")
    df = pd.read_csv('mappings.csv')
    process_jsondata_gene("data/TBGA/test1.json", classes_ogg, 20516)  # Train 178264 | val 20193 | test 20516
    process_jsondata_dis("data/TBGA/filenew.json", df, 20516)


    extractProcessedId("data/TBGA/filenew.json", "data/TBGA/processed_test.json", 20516)

    createSubset("data/TBGA/TBGA_test_short.json", "NA", targetdir="data/TBGA/newsubset.json", total=6500)
    replacespecial('data/TBGA/TBGA_train_processed_subset.json', "data/TBGA/subset.json", 43428)
    re_print("data/TBGA/newsubset.json", 'data/TBGA/TBGA_test_512.json')

if __name__ == '__main__':
    #main()
    directory = "data/TBGA/TBGA_short.json"
    check_sentence_length("data/TBGA/TBGA_test_processed.json")
    re_print("data/TBGA/TBGA_shorttextset.json", "data/TBGA/TBGA_short.json")
    countclasses(directory)