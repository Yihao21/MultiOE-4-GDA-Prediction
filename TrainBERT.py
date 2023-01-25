import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from pytorch_pretrained_bert import BertTokenizer
from TBGADataset import TBGADataset, process_txtdata
from model import MultiBert, ERNIEModel
from configs.config import DefaultConfig as cfg
from transformers import get_linear_schedule_with_warmup
from utils import trainigprocessBERT, load_model, evalBERT
import wandb
import logging

logging.basicConfig(filename="./Experiment.txt", filemode="w", level=logging.INFO, format='%(message)s')


if __name__ == '__main__':
    print("Using GPU: {}".format(torch.cuda.is_available()))
    torch.cuda.empty_cache()
    wandb.init(name="TBGA128_BERT_testonDG", project="SCAI_MULTIONTO")

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # train_dir = "data/TBGA/TBGA_train_processed_subset.json"
    # #train_dir = "data/DG_train.json"
    # train_data, train_label, _, _ = process_txtdata(train_dir)
    # training = list(zip(train_data, train_label))
    #
    # valid_dir = "data/TBGA/TBGA_val_processed.json"
    # #valid_dir = "data/DG_valid.json"
    # val_data, val_label, _, _ = process_txtdata(valid_dir)
    # validation = list(zip(val_data, val_label))
    #
    # print("Process Data---------------------------------")
    # train_dataset = TBGADataset(training, bert_tokenizer)
    # print("Training set processed")
    # valid_dataset = TBGADataset(validation, bert_tokenizer)
    #
    # print("Valid and Test set processed")
    #
    # train_dataloader = DataLoader(dataset=train_dataset, sampler=RandomSampler(train_dataset), batch_size=16,
    #                               drop_last=True)
    # validation_dataloader = DataLoader(dataset=valid_dataset, sampler=RandomSampler(valid_dataset), batch_size=16,
    #                                    drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # # Training
    model = MultiBert(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)


    # epochs 2-4
    # epochs = 3
    #
    # # Total number of training steps is [number of batches] x [number of epochs].
    # # (Note that this is not the same as the number of training samples).
    # total_steps = len(train_dataloader) * epochs
    #
    # # Create the learning rate scheduler.
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
    #
    # trainigprocessBERT(epochs, model, optimizer, scheduler, train_dataloader, validation_dataloader, device)


    # EVAL----------------------------------------------
    model, optimizer = load_model(model, optimizer, "./models/BERT_checkpoint_epoch_3.pth")
    #test_dir = "data/TBGA/TBGA_test_processed.json"
    test_dir = "data/DG_test.json"
    test_data, test_label, _, _ = process_txtdata(test_dir)
    testing = list(zip(test_data, test_label))
    test_dataset = TBGADataset(testing, bert_tokenizer)

    test_dataloader = DataLoader(dataset=test_dataset, sampler=SequentialSampler(test_dataset), batch_size=16,
                                 drop_last=True)

    evalBERT(model, test_dataloader, device)


    wandb.finish()