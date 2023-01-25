import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import json
import gensim
from transformers import BertTokenizerFast
from spacy.lang.en import English
from spacy.training import offsets_to_biluo_tags
from configs.config import cfg
from tqdm import tqdm
import logging
import numpy as np

logging.getLogger("spacy")
logging.basicConfig(filemode="w", filename="LOG.txt", level=logging.WARNING)


'''
Author: Yihao Wang@Fraunhofer SCAI
Part of code was referenced from Phillip Wegner@Fraunhofer SCAI's work
'''

relations2id = {"NA": 0, "therapeutic": 1, "biomarker": 2, "genomic_alterations": 3}
#relations2id = {"NA": 0, "therapeutic": 1, "biomarker": 1, "genomic_alterations": 1}
#relations2id = {"no": 0, "yes": 1}

# process the TBGA.json dataset
def process_txtdata(dir):
    gene = {"id":[], "name":[], "pos":[]}
    disease = {"id":[],"name":[], "pos":[]}
    # old version of reading dataset
    # with open(dir,"r") as f:
    #     for line in f:
    #         j_content = json.loads(line)
    #         text_seq.append(j_content['text'])
    #         labels.append(relations2id[j_content["relation"]])
    #         gene["id"].append(j_content["h"]["id"])
    #         #gene["name"].append(j_content["h"]["name"])
    #         gene["pos"].append(j_content["h"]["pos"])
    #         disease["id"].append(j_content["t"]["id"])
    #         #disease["name"].append(j_content["t"]["name"])
    #         disease["pos"].append(j_content["t"]["pos"])
    # f.close()
    with open(dir, "r") as f:
        data = json.load(f)
    text_seq = [data[idx]["text"] for idx in range(len(data))]
    labels = [relations2id[data[idx]["relation"]] for idx in range(len(data))]
    gene_id = [data[idx]["h"]["id"] for idx in range(len(data))]
    gene_name = [data[idx]["h"]["name"] for idx in range(len(data))]
    gene_pos = [data[idx]['h']["pos"] for idx in range(len(data))]
    gene = dict(id=gene_id, name=gene_name, pos=gene_pos)
    dis_id = [data[idx]["t"]["id"] for idx in range(len(data))]
    dis_name = [data[idx]["t"]["name"] for idx in range(len(data))]
    dis_pos = [data[idx]['t']["pos"] for idx in range(len(data))]
    disease = dict(id=dis_id, name=dis_name, pos=dis_pos)

    return text_seq, labels, gene, disease

class TBGADataset(Dataset):
    def __init__(self, dataset, bert_tokenizer, gene_ann=None, dis_ann=None, ADD_KNOWLEDGE=False, nlp=None,SINGLE_ONTO=False):
        '''
        :param dataset:
        :param bert_tokenizer:
        :param ADD_KNOWLEDGE:
        :param ruler:            along with add knowledge
        '''
        self.cfg = cfg
        self.bert_tokenizer = bert_tokenizer
        self.dataset = dataset
        self.ADD_KNOWLEDGE = ADD_KNOWLEDGE
        self.gene_ann = gene_ann
        self.dis_ann = dis_ann
        self.SINGLE_ONTO=SINGLE_ONTO
        self.max_seq = 128
        if self.ADD_KNOWLEDGE:
            self.nlp = nlp
            self.embed = gensim.models.Word2Vec.load("embeddings/oggdoid120/output.embeddings")
            tokens_seq, att_mask, labels, embed_ents, ent_mask = self.get_input_knowledge(self.dataset, self.gene_ann, self.dis_ann, self.nlp, self.embed, max_seq_length=self.max_seq,SINGLE_ONTO=self.SINGLE_ONTO)
            self.data = torch.tensor(tokens_seq, dtype=torch.long)
            self.att_msk = torch.tensor(att_mask, dtype=torch.long)
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.ents = torch.stack(embed_ents)
            self.ent_msk = torch.tensor(ent_mask, dtype=torch.long)

        else:
            seqs, seqs_msk, labels = self.get_input(self.dataset, self.max_seq)
            self.data = torch.tensor(seqs, dtype=torch.long)
            self.mask = torch.tensor(seqs_msk, dtype=torch.long)
            self.labels = torch.tensor(labels, dtype=torch.long)


    # Get tag using offsets_to_biluo_tags
    def get_iobtags(self, nlp, sentence, ann_gene, ann_disease, idx, SINGLE_ONTO):
        if SINGLE_ONTO:
            #entities = [(ann_disease["pos"][idx][0], ann_disease["pos"][idx][0] + ann_disease["pos"][idx][1], ann_disease["id"][idx])]
            entities = [(ann_gene["pos"][idx][0], ann_gene["pos"][idx][0] + ann_gene["pos"][idx][1], ann_gene["id"][idx])]
        else:
            entities = [(ann_gene["pos"][idx][0], ann_gene["pos"][idx][0] + ann_gene["pos"][idx][1], ann_gene["id"][idx]),
                       (ann_disease["pos"][idx][0], ann_disease["pos"][idx][0] + ann_disease["pos"][idx][1], ann_disease["id"][idx])]
        #print(sentence[ann_gene["pos"][idx][0] : ann_gene["pos"][idx][0] + ann_gene["pos"][idx][1]])
        #print(sentence[ann_disease["pos"][idx][0] : ann_disease["pos"][idx][0] + ann_disease["pos"][idx][1]])
        doc = nlp(sentence)
        try:
            # BILUO(Begin, In, Last, Unit, Out)
            tags = [str(tag).replace("L-", "I-").replace("U-","B-").replace("I-","B-") for tag in offsets_to_biluo_tags(doc, entities)]
        except:
            print("Problem text " + sentence)
            raise
        return tags

    # Tags has the length of sentence(split by blank space), normally we will have longer sequence after tokenize
    # This function expand the tags corresponding to the tokenized sequence which also preserve the labels
    def expandtags(self, tags, tokenizedid):
        tags.extend(["0"] * (len(tokenizedid) - len(tags) - 2))              # Watch out when modify tags, the element is string
        new_tags = []
        word_idx = 0
        duplicate = False
        for i in range(1, len(tokenizedid) - 1):
            if tokenizedid[i] == word_idx:
                new_tags.append(tags[word_idx])
                if not duplicate:  duplicate = True
            else:
                word_idx += 1
                duplicate = False
                new_tags.append(tags[word_idx])

        # add 0 for cls and sep
        new_tags.append("0")
        new_tags.insert(0, "0")
        return new_tags

    def get_input_knowledge(self, dataset, gene_label, disease_label, nlp, embed, max_seq_length, SINGLE_ONTO=False):
        # unzip the dataset
        sentences = [i[0] for i in dataset]
        labels = [i[1] for i in dataset]

        embed_ents = []
        ent_mask = []
        tokens_seq = []
        attention_msk_seq = []

        for i, sentence in enumerate(tqdm(sentences)):
            # print("Process Sentence {}: {}".format(i, sentence))
            encoding = self.bert_tokenizer(sentence)                # each element has type of BatchEncoding, which is a dict of tokenized2id, tokentypeid and attention mask
            tag_label = self.get_iobtags(nlp=nlp, sentence=sentence, ann_gene=gene_label, ann_disease=disease_label, idx=i, SINGLE_ONTO=SINGLE_ONTO)
            expand_tags = self.expandtags(tag_label, encoding.word_ids())    # encoding.words
            ents = []
            msk = []

            # print("Length of sentence: {}".format(len(sentence.split(" "))))
            # print("Bert tokenizer ({}): ".format(len(encoding["input_ids"])))
            # print(encoding["input_ids"])
            # print(encoding.tokens())
            # print("Spacy tagger ({}): ".format(len(tag_label)))
            # print(tag_label)
            # print("Expanded Tags ({}): ".format(len(expand_tags)))
            # print(expand_tags)

            '''
            For every expanded tagged sentences, if one element is (part of) an entity, we replace it with the embedding vector dimension(100),
            otherwise a zero vector with dimension(100).
            '''
            for idx in range(len(expand_tags)):
                if expand_tags[idx].startswith("B-"):
                    ents.append(embed.wv.get_vector(expand_tags[idx].replace("B-", "")).tolist())
                else:
                    ents.append([0] * self.cfg.EMBEDDING_DIMENSION)

            att_msk = len(ents)*[1]
            # print("old length: {}".format(len(ents)))
            curr_seq_length = len(ents)
            if curr_seq_length <= max_seq_length:
                ents.extend([[0] * self.cfg.EMBEDDING_DIMENSION] * (max_seq_length - curr_seq_length))
                att_msk.extend([0] * (max_seq_length - curr_seq_length))
            else:
                ents = ents[0: max_seq_length]
                att_msk = att_msk[0: max_seq_length]
            # print("new length: {}".format(len(ents)))

            #att_msk_as_tensor = torch.tensor(att_msk)
            attention_msk_seq.append(att_msk)

            ents_as_tensor = torch.tensor(ents, dtype=torch.float)
            embed_ents.append(ents_as_tensor)
            for emb in ents:
                if emb != [0] * self.cfg.EMBEDDING_DIMENSION:
                    msk.append(1)
                else:
                    msk.append(0)

            ent_mask.append(msk)

            tokens = self.trunate_and_pad(encoding["input_ids"], max_seq_length)
            tokens_seq.append(tokens)
            #label_seq.append([labels[i]] * max_seq_length)


        return tokens_seq, attention_msk_seq, labels, embed_ents, ent_mask

    def trunate_and_pad(self, seq, max_seq_len):
        if len(seq) > (max_seq_len):
            new_seq = seq[0: (max_seq_len)]
        else:
            new_seq = seq + [0] * (max_seq_len - len(seq))
        return new_seq


    # This function used for the task without adding any prior knowledge
    # also the return value is adjusted for Bert, Biobert etc.
    def get_input(self, dataset, max_seq_len):
        sentences = [i[0] for i in dataset]
        labels = [i[1] for i in dataset]

        tokens_seq = []
        for i in range(len(sentences)):
            tokens_seq.append(self.bert_tokenizer.tokenize(sentences[i]))

        seqs = []
        seq_masks = []
        for t_s in tokens_seq:
            sq, sq_m = self.process_sequence(t_s, max_seq_len)
            seqs.append(sq)
            seq_masks.append(sq_m)

        return seqs, seq_masks, labels


    # trunate and padding manually, this function was called by the function get_input
    def process_sequence(self, seq, max_seq_len):
        if len(seq) > (max_seq_len - 2):
            seq = seq[0: (max_seq_len - 2)]

        seq = ['[CLS]'] + seq + ['[SEP]']
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)

        padding = [0] * (max_seq_len - len(seq))

        # attention_mask , which is a binary mask that identifies whether a token is a real word or just padding.
        seq_mask = [1] * len(seq) + padding

        # seq_segment = [0] * len(seq) + padding

        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        # assert len(seq_segment) == max_seq_len
        return seq, seq_mask

    def __getitem__(self, idx):
        if self.ADD_KNOWLEDGE:
            return self.data[idx], self.att_msk[idx], self.labels[idx], self.ents[idx], self.ent_msk[idx]
        return self.data[idx], self.mask[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    # val_dir = "data/TBGA/val.json"
    # val_data, val_label, gene_ann_val, dis_ann_val = process_txtdata(val_dir)
    # val_dataset = list(zip(val_data, val_label))
    # bert_tokenizer_fast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    #
    # nlp = English()
    # nlp.tokenizer.from_disk("tokenizer/tokenizer")
    # ruler = nlp.add_pipe("entity_ruler")
    # ruler = ruler.from_disk("ruler/ogg_doid")
    #
    # val_set = TBGADataset(val_dataset, bert_tokenizer_fast, gene_ann_val, dis_ann_val, ADD_KNOWLEDGE=True, nlp=nlp)
    # val_dataloader = DataLoader(dataset=val_set, sampler=RandomSampler(val_set), batch_size=4, drop_last=True)
    #
    #
    # data, label, ent, ent_msk = next(iter(val_dataloader))
    #example = "The TEL/ARG oncogene is formed by t(1;12)(q25;p13) reciprocal translocation and is associated with human leukemia."
    #print(example[8:11])
    # print(data)
    # print(label)
    # print(ent)
    # print(ent_msk)


    # Utils: Count the number of samples
    data_dir = "data/TBGA/TBGA_train_processed_subset.json"
    _, label, _, _ = process_txtdata(data_dir)
    count_class = {"NA": 0, "therapeutic": 0, "biomarker": 0, "genomic_alterations": 0}
    for l in label:
        if l == 0: count_class["NA"] += 1
        elif l == 1: count_class["therapeutic"] += 1
        elif l == 2: count_class["biomarker"] += 1
        elif l == 3: count_class["genomic_alterations"] += 1
    for k in count_class.keys():
        print('{} has {} samples. {:.4f}'.format(k, str(count_class[k]), count_class[k]/len(label)))

    print('Number of samples: {}'.format(len(label)))



    # dataset = list(zip(data, label))
    # bert_tokenizer_fast = BertTokenizerFast.from_pretrained("dmis-lab/biobert-v1.1")
    #
    # nlp = English()
    # ruler = nlp.add_pipe("entity_ruler")
    # ruler = ruler.from_disk("ruler/go_doid")
    #
    # TBGADataset = TBGADataset(dataset, bert_tokenizer_fast, ADD_KNOWLEDGE=False, nlp=nlp)


