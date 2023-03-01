import json
from tqdm import tqdm

relations2id = {"NA": 0, "therapeutic": 1, "biomarker": 2, "genomic_alterations": 3}

def process_txtdata(dir):
    with open(dir, "r") as f:
        data = json.load(f)

    labels = [relations2id[data[idx]["relation"]] for idx in range(len(data))]

    gene_id = [data[idx]["h"]["id"] for idx in range(len(data))]
    dis_id = [data[idx]["t"]["id"] for idx in range(len(data))]

    return labels, gene_id, dis_id

def find_tuple(labels, gene_id, dis_id):
    bridge_tuples = {}        # key: gene-dis tuple, value: number of times they shown in the dataset
    for idx, l in enumerate(tqdm(labels)):
        if l == 0:
            continue
        gene_dis = (gene_id[idx], dis_id[idx])
        if gene_dis in bridge_tuples:   bridge_tuples[gene_dis] += 1
        else: bridge_tuples[gene_dis] = 1

    return bridge_tuples


if __name__ == '__main__':
    labels, gene, dis = process_txtdata("data/TBGA/TBGA_train_processed.json")
    bridge_tuples = find_tuple(labels, gene, dis)
    #sort_by_value = dict(sorted(bridge_tuples.items(), key=lambda item: item[1]))
    for k,v in bridge_tuples.items():
        if v >= 15: print("Tuple {} show {} times".format(k, v))


