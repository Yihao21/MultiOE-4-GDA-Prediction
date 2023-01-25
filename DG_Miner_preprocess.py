import pandas as pd
from tqdm import tqdm
import json
import csv
import random

'''
This function create "TBGA" style dataset
{Sentences: XXX, relation:"yes/no", t:{"disease iri", pos}, h:{"gene iri", posiri}}
'''
def create_dataset(df):
    dataset = []
    dis_name = df['Disease']
    dis_label = df['Diseaseiri']
    gene_name = df['Gene']
    gene_label = df['Geneiri']

    for idx in range(len(dis_name)):
        sentence_idx = "Is {} related to {}?".format(dis_name[idx], gene_name[idx])
        new_disid = "http://purl.obolibrary.org/obo/" + str(dis_label[idx]).replace(":","_")
        new_geneid = "http://purl.obolibrary.org/obo/" + str(gene_label[idx]).replace(":","_")
        t_idx = dict(id=new_disid, name=dis_name[idx], pos=[3, len(dis_name[idx])])
        h_idx = dict(id=new_geneid, name=gene_name[idx], pos=[15 + len(dis_name[idx]), len(gene_name[idx])])    #print(test[3 + len("addd") + 12:])
        da = dict(text=sentence_idx, relation="yes", t=t_idx, h=h_idx)
        dataset.append(da)

    with open("DG_dataset.json", "w") as fw:
        json.dump(dataset, fw)

    fw.close()

# create a dictonary of dis-gene pair, since there is duplicate in disease, so cannot use zip() instead
# used later for create non-related pair.
def create_dis_gen_dic(df):
    dis_gene_pair = {}
    dis_name = df['Disease']
    gene_name = df['Gene']
    for idx, dis in enumerate(tqdm(dis_name)):
        if dis in dis_gene_pair:
            dis_gene_pair[dis].append(gene_name[idx])
        else:
            dis_gene_pair[dis] = [gene_name[idx]]

    return dis_gene_pair

# Create negative examples:
def create_neg(df):
    dis_gene_pair = create_dis_gen_dic(df)
    '''TO do what does gene looks like
    '''
    neg_samples = []
    dis_name = df['Disease']
    dis_label = df['Diseaseiri']
    gene_name = df['Gene']
    gene_label = df['Geneiri']
    total = len(dis_name)

    for idx in range(total):

        current_disname = dis_name[idx]
        new_disid = "http://purl.obolibrary.org/obo/" + str(dis_label[idx]).replace(":","_")
        random_idx = random.randint(0, total-1)
        neg_gene = gene_name[random_idx]
        # If the relation already exists
        if neg_gene in dis_gene_pair[current_disname]:
            continue
        else:
            sentence_idx = "Is {} related to {}?".format(current_disname, neg_gene)
            new_geneid = "http://purl.obolibrary.org/obo/" + str(gene_label[random_idx]).replace(":","_")
            t_idx = dict(id=new_disid, name=current_disname, pos=[3, len(current_disname)])
            h_idx = dict(id=new_geneid, name=neg_gene, pos=[15 + len(current_disname), len(neg_gene)])
            da = dict(text=sentence_idx, relation="no", t=t_idx, h=h_idx)
            neg_samples.append(da)

    return neg_samples

def append_negsamples(df, dir):
    neg = create_neg(df)
    with open(dir) as f:
        data = json.load(f)
    f.close()
    data.extend(neg)

    with open("DG_dataset_posandneg.json", "w") as fw:
        json.dump(data, fw)

    fw.close()

# Take out all referenceable diease
def mapping_diseae(dis_processed, dis_map_processed, mappingref, gene):
    '''

    :param dis_processed: list: Mesh code of disease in DG-Miner(without "Mesh:")
    :param dis_map_processed: list: Mesh code of disease cross reference of Mesh code and DOID iri(without "Mesh:")
    :param mappingref: dataframe: mapping table include iri, name, meshcode etc.
    :param gene: list: Uniprot code of gene
    :return: None. but save a dataframe to csv
    '''
    dis_iri = []
    dis_name = []
    gene_iri = []
    notfound = 0
    gene_name = []
    for idx, dis_found in enumerate(tqdm(dis_processed)):
        try:
            idx_found = dis_map_processed.index(dis_found)
            dis_iri.append(mappingref["curie_id"][idx_found])
            dis_name.append(mappingref["label"][idx_found])
            gene_iri.append(gene[idx])                 # here index is from the disgenet dataset
            gene_name.append("None")
        except:
            notfound += 1

    df = pd.DataFrame(zip(dis_iri, dis_name, gene_iri, gene_name), columns=["Diseaseiri", "Disease", 'Geneiri', "Gene"])
    df.to_csv('DGdata.csv', index=False)
    print("Number of not matching disease {}".format(notfound))

def mapping_gene2HGNC(dgdataframe, mappinghgnc):
    gene = dgdataframe['Geneiri']
    hcbi = list(mappinghgnc['HGNC ID'])
    uniprot = list(mappinghgnc['UniProt ID(supplied by UniProt)'])

    gene_iri = []
    gene_name = []
    dis_iri = []
    dis_name = []
    notfound = 0

    for i, gene_found in enumerate(tqdm(gene)):
        try:
            idx_found = uniprot.index(gene_found)
            matched_hcbi = hcbi[idx_found]
            gene_iri.append(matched_hcbi)
            gene_name.append("None")
            dis_iri.append(dgdataframe["Diseaseiri"][i])
            dis_name.append(dgdataframe["Disease"][i])
        except:
            notfound += 1

    df = pd.DataFrame(zip(dis_iri, dis_name, gene_iri, gene_name), columns=["Diseaseiri", "Disease", 'Geneiri', "Gene"])
    df.to_csv('DGdata_HGNC.csv', index=False)
    print("Number of not matching gene {}".format(notfound))

def mapping_HGNC2ogg(dgdataframe, mappingogg):
    hgnc_id = list(mappingogg["mapped_curie"])
    oggiri = list(mappingogg["curie_id"])
    oggname = list(mappingogg["label"])
    gene = dgdataframe['Geneiri']

    gene_iri = []
    gene_name = []
    dis_iri = []
    dis_name = []
    notfound = 0

    for i, gene_found in enumerate(tqdm(gene)):
        try:
            idx_found = hgnc_id.index(gene_found)
            matched_hcbi = oggiri[idx_found]
            gene_iri.append(matched_hcbi)
            gene_name.append(oggname[idx_found])
            dis_iri.append(dgdataframe["Diseaseiri"][i])
            dis_name.append(dgdataframe["Disease"][i])
        except:
            notfound += 1

    df = pd.DataFrame(zip(dis_iri, dis_name, gene_iri, gene_name), columns=["Diseaseiri", "Disease", 'Geneiri', "Gene"])
    df.to_csv('DGdata_final.csv', index=False)
    print("Number of not matching gene {}".format(notfound))

def createSubset(dir, targetdir):
    new_data=[]
    with open(dir) as f:
        data = json.load(f)
    total = len(data)
    for idx in tqdm(range(total)):

        label = data[idx]["relation"]
        if label == "yes":
            number = random.randint(0, 100)
            if number > 90:
                new_data.append(data[idx])
        else:
            number = random.randint(0, 100)
            if number > 70:
                new_data.append(data[idx])

    print("New data size {}".format(len(new_data)))
    with open(targetdir, "w") as fw:
        json.dump(new_data, fw)

    fw.close()

def splitdataset(dir):
    with open(dir) as f:
        data = json.load(f)

    random.shuffle(data)
    total = len(data)
    train_len = int(0.7 * total)
    valid_len = int(0.2 * total)
    test_len = total - train_len - valid_len
    assert total == (train_len+valid_len+test_len)


    train_set = data[:train_len]
    valid_set = data[train_len:train_len + valid_len]
    test_set = data[train_len + valid_len:]
    with open("DG_train_org.json", "w") as fw:
        json.dump(train_set, fw)
    with open("DG_valid_org.json", "w") as fw2:
        json.dump(valid_set, fw2)
    with open("DG_test_org.json", "w") as fw3:
        json.dump(test_set, fw3)
    fw.close()
    fw2.close()
    fw3.close()



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

if __name__ == '__main__':
    splitdataset("./DG_dataset_sub.json")
    #createSubset("./DG_dataset_posandneg.json", "./DG_dataset_sub.json")

    # # Mapping disease Mesh <-> DOID iri
    # df1 = pd.read_csv('DG-Miner_miner-disease-gene.tsv', sep='\t')       #MESH
    # dis = df1['# Disease(MESH)']
    # gene = df1['Gene']
    # dis_processed = [d[5:] for d in dis]        #address name of mesh is different stored
    # mappingref = pd.read_csv('doidmeshmappings.tsv', sep='\t')           #MeSH
    # dis_map = mappingref['mapped_curie']
    # dis_map_processed = [d[5:] for d in dis_map]
    # mapping_diseae(dis_processed, dis_map_processed, mappingref, gene)

    # # Mapping Gene UniProt <-> HGNC
    # df1 = pd.read_csv('DGdata.csv')
    # mappinghgnc = pd.read_csv('HGNC UNIPROTmapping.tsv', sep='\t')
    # mapping_gene2HGNC(df1, mappingogg)

    # Mapping Gene HGNC <-> OGG iri
    # df1 = pd.read_csv('DGdata_HGNC.csv')
    # mappingogg = pd.read_csv('mappingsHGNCOGG.csv')
    # mapping_HGNC2ogg(df1, mappingogg)

    #df1 = pd.read_csv('DGdata_final.csv')
    #create_dataset(df1)


    # # Test

    # diseaseiri = list(df1['Diseaseiri'])
    # mappingref = pd.read_csv('doidmeshmappings.tsv', sep='\t')           #MeSH
    # dis_map = list(mappingref['mapped_curie'])
    # dis_map_iri = list(mappingref['curie_id'])
    # dis_map_processed = [d[5:] for d in dis_map]
    # idx = 1829
    # d = diseaseiri[idx]
    # print("Disease IRI: {}".format(d))
    # mesh_idx = dis_map_iri.index(d)
    # d_mesh = dis_map[mesh_idx]
    # print("Disease Mesh: {}".format(d_mesh))
    #
    # geneiri = list(df1["Geneiri"])
    # print(geneiri[idx])
    # mappingogg = pd.read_csv('mappingsHGNCOGG.csv')
    # ogg_iri = list(mappingogg['curie_id'])
    # ogg_hgnc = list(mappingogg['mapped_curie'])
    # ogg_idx = ogg_iri.index(geneiri[idx])
    # a = ogg_hgnc[ogg_idx]
    # print(ogg_hgnc[ogg_idx])
    #
    # mappinghgnc = pd.read_csv('HGNC UNIPROTmapping.tsv', sep='\t')
    # ogg_map_hgnc = list(mappinghgnc['HGNC ID'])
    # ogg_map_uniprot = list(mappinghgnc['UniProt ID(supplied by UniProt)'])
    # ogg_map_idx = ogg_map_hgnc.index(a)
    # gene_uniprot = ogg_map_uniprot[ogg_map_idx]
    # print("Gene Uniprot: {}".format(gene_uniprot))
    #
    # dgminer = pd.read_csv('DG-Miner_miner-disease-gene.tsv', sep='\t')
    # gene_dgminer = list(dgminer['Gene'])
    # dis_dgminer = list(dgminer['# Disease(MESH)'])
    # for idx, g in enumerate(tqdm(gene_dgminer)):
    #     if g == gene_uniprot:
    #         d = dis_dgminer[idx]
    #         print(d)

    # dgp = create_dis_gen_dic(df1)
    # for k, v in dgp.items():
    #     print(k)
    #     print(v)
    #     print("*********************************************")

    #re_print("./DG_dataset_sub.json", "DG_sub.json")
    re_print("./DG_train_org.json","./DG_train.json")
    re_print("./DG_valid_org.json", "./DG_valid.json")
    re_print("./DG_test_org.json", "./DG_test.json")








