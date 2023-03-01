import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from transformers import BertTokenizerFast
from spacy.lang.en import English
from TBGADataset import TBGADataset, process_txtdata
from model import ERNIEModel
from configs.config import DefaultConfig as cfg
from transformers import get_linear_schedule_with_warmup
from utils import trainigprocessERNIE, load_model, evalERNIE
import wandb
import logging

logging.basicConfig(filename="./ExperimentERNIE.txt", filemode="w", level=logging.INFO, format='%(message)s')


if __name__ == '__main__':
    print("Using GPU: {}".format(torch.cuda.is_available()))
    torch.cuda.empty_cache()
    wandb.init(name="TBGA128_4cls_multiemb120_traintest", project="SCAI_MULTIONTO")   # DG_ERNIE_MergedEmbsub_2cls_testonTBGA_short_nopretrain


    bert_tokenizer_fast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    nlp = English()
    ruler = nlp.add_pipe("entity_ruler")
    ruler = ruler.from_disk("ruler/ogg_doid1010")
    nlp.tokenizer.from_disk("tokenizer/tokenizer")

    ### Training

    train_dir = "dataset/TBGA/TBGA_train_processed.json"       #  | "dataset/DG/DG_train.json"
    train_data, train_label, gene_ann, dis_ann = process_txtdata(train_dir)
    train_dataset = list(zip(train_data, train_label))

    val_dir = "dataset/TBGA/TBGA_val_processed.json"         #  | "dataset/DG/DG_valid.json"
    val_data, val_label, gene_ann_val, dis_ann_val = process_txtdata(val_dir)
    val_dataset = list(zip(val_data, val_label))

    print("Process Data---------------------------------")
    training_set = TBGADataset(train_dataset, bert_tokenizer_fast, gene_ann, dis_ann, ADD_KNOWLEDGE=True, nlp=nlp, SINGLE_ONTO=False)
    print("Training set processed")
    val_set = TBGADataset(val_dataset, bert_tokenizer_fast, gene_ann_val, dis_ann_val, ADD_KNOWLEDGE=True, nlp=nlp, SINGLE_ONTO=False)
    print("Validation set processed")

    train_dataloader = DataLoader(dataset=training_set, sampler=RandomSampler(training_set), batch_size=16,
                                  drop_last=True)
    val_dataloader = DataLoader(dataset=val_set, sampler=RandomSampler(val_set), batch_size=16, drop_last=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ### Training
    model = ERNIEModel(config=cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

    # epochs 2-4
    epochs = 3

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    trainigprocessERNIE(epochs, model, optimizer, scheduler, train_dataloader, train_dataloader, device)

    ### Evaluation
    #model, optimizer = load_model(model, optimizer, "./models/ERNIE_checkpoint_epoch_3.pth")
    test_dir = "dataset/TBGA/TBGA_test_processed.json"  # | "dataset/DG/DG_test.json"
    test_data, test_label, gene_ann_test, dis_ann_test = process_txtdata(test_dir)
    test_dataset = list(zip(test_data, test_label))

    test_set = TBGADataset(test_dataset, bert_tokenizer_fast, gene_ann_test, dis_ann_test, ADD_KNOWLEDGE=True, nlp=nlp, SINGLE_ONTO=True)
    test_dataloader = DataLoader(dataset=test_set,batch_size=16, drop_last=True, shuffle=True)
    evalERNIE(model, test_dataloader, device)


    wandb.finish()



    # Single value test
    # data, att_msk, label, ent, ent_msk = next(iter(test_dataloader))
    # output,_ = model(data, att_msk, ent, ent_msk, label)
    # print(output)


