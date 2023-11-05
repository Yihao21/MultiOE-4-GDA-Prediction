import numpy as np
import os
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix
import wandb
from configs.config import cfg
import logging
import itertools

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat), pred_flat, labels_flat

def save_model(model, optimizer, epoch, name):
    """ Saving model checkpoint """

    if (not os.path.exists("models")):
        os.makedirs("models")
    savepath = f"models/{name}_checkpoint_epoch_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, savepath)
    return

def load_model(model, optimizer, savepath):
    """ Loading pretrained checkpoint """

    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint["epoch"]
    #stats = checkpoint["stats"]
    return model, optimizer

def trainigprocessBERT(epochs, model, optimizer, scheduler, train_dataloader, validation_dataloader, device):
    for epoch_i in range(0, epochs):

        total_train_loss = 0
        running_loss = 0

        model.train()

        for step, (seq, msk, labels) in enumerate(tqdm(train_dataloader)):
            b_input_ids = seq.to(device)
            b_input_mask = msk.to(device)
            b_labels = labels.to(device)

            model.zero_grad()

            output, loss = model(input_ids=b_input_ids,attention_mask=b_input_mask,labels=b_labels)

            total_train_loss += loss.item() * output.size(0)
            running_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

            if step % cfg.print_every == 0:
                wandb.log({'batch_loss':  running_loss/cfg.print_every})
                running_loss = 0

        avg_train_loss = total_train_loss / len(train_dataloader)

        wandb.log({'training_loss': avg_train_loss/len(train_dataloader)})
        save_model(model, optimizer, epoch_i + 1, "BERT")

        ### Validation
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        f1_macro_list = []
        recall_list = []
        precision_list = []
        rocauc_list = []

        for (seq, msk, labels) in tqdm(validation_dataloader):
            b_input_ids = seq.to(device)
            b_input_mask = msk.to(device)
            b_labels = labels.to(device)

            with torch.no_grad():
                preds, loss = model(input_ids=b_input_ids,attention_mask=b_input_mask,labels=b_labels)

            total_eval_loss += loss.item()

            pred_prob = preds.softmax(dim=1).detach().cpu().numpy()

            preds = preds.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            acc, pre, gt = flat_accuracy(preds, label_ids)
            total_eval_accuracy += acc

            precision_list.append(precision_score(gt, pre, average='micro'))
            r_s = recall_score(gt, pre, average='macro')
            recall_list.append(r_s)
            f1_s = f1_score(gt, pre, average='macro')
            f1_macro_list.append(f1_s)
            try:
                rocauc = roc_auc_score(y_true=pre, y_score=pred_prob, multi_class='ovr')
                rocauc_list.append(rocauc)
            except ValueError:
                logging.warning("Some class not shown in the mini-batch")
            wandb.log({'Accuracy': acc, "Recall Score": r_s, "F1 Macro Score": f1_s})


        avg_val_loss = total_eval_loss / len(validation_dataloader)


def evalBERT(model, test_dataloader, device):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    f1_macro_list = []
    recall_list = []
    precision_list = []
    rocauc_list = []
    y_true = []
    y_pred = []

    # Evaluate data for one epoch
    for (seq, msk, labels) in tqdm(test_dataloader):
        b_input_ids = seq.to(device)
        b_input_mask = msk.to(device)
        b_labels = labels.to(device)

        with torch.no_grad():
            preds, loss = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)

        total_eval_loss += loss.item()

        pred_prob = preds.softmax(dim=1).detach().cpu().numpy()

        # Move logits and labels to CPU
        preds = preds.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        acc, pre, gt = flat_accuracy(preds, label_ids)
        total_eval_accuracy += acc

        precision_list.append(precision_score(gt, pre, average='micro'))
        r_s = recall_score(gt, pre, average='macro')
        recall_list.append(r_s)
        f1_s = f1_score(gt, pre, average='macro')
        f1_macro_list.append(f1_s)
        try:
            rocauc = roc_auc_score(y_true=pre, y_score=pred_prob, multi_class='ovr')
            rocauc_list.append(rocauc)
        except ValueError:
            logging.warning("Some class not shown in the mini-batch")
        wandb.log({'Test Accuracy': acc, "Test Recall Score": r_s, "Test F1 Macro Score": f1_s})
        y_true.append(gt)
        y_pred.append(pre)

    y_true_m = list(itertools.chain.from_iterable(y_true))
    y_pred_m = list(itertools.chain.from_iterable(y_pred))
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                       preds=y_true_m, y_true=y_pred_m,
                                                       class_names=["NA", "therapeutic", "biomarker", "genomic_alterations"])})

    try:
        wandb.log({'ROC AUC Score': sum(rocauc_list) / len(rocauc_list)})
    except:
        logging.info(" ROC AUC Score cannot be calculated because of divide by zero")

def trainigprocessERNIE(epochs, model, optimizer, scheduler, train_dataloader, val_dataloader, device):
    for epoch_i in range(0, epochs):
        total_train_loss = 0
        running_loss = 0

        model.train()

        # For each batch of training data...
        for step, (data, att_msk, label, ents, ent_msk) in enumerate(tqdm(train_dataloader)):
            data, att_msk, label, ent, ent_msk = data.to(device), att_msk.to(device), label.to(device), ents.to(device), ent_msk.to(device)

            model.zero_grad()

            output, loss = model(data, att_msk, ent, ent_msk, label)  # BSZ, 64, num_labels

            total_train_loss += loss.item() * output.size(0)
            running_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

            if step % cfg.print_every == 0:
                wandb.log({'batch_loss':  running_loss/cfg.print_every})
                running_loss = 0

        avg_train_loss = total_train_loss / len(train_dataloader)
        wandb.log({'training_loss': avg_train_loss / len(train_dataloader)})

        save_model(model, optimizer, epoch_i + 1, "ERNIE")

        ### Validation
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        f1_macro_list = []
        recall_list = []
        precision_list = []
        rocauc_list = []

        # Evaluate data for one epoch
        for (data, att_msk, label, ents, ent_msk) in tqdm(val_dataloader):
            data, att_msk, label, ent, ent_msk = data.to(device), att_msk.to(device), label.to(device), ents.to(
                device), ent_msk.to(device)

            with torch.no_grad():
                preds, loss = model(data, att_msk, ent, ent_msk, label)  # BSZ, 64, num_labels

            total_eval_loss += loss.item()

            pred_prob = preds.softmax(dim=1).detach().cpu().numpy()

            # Move logits and labels to CPU
            preds = preds.detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()

            acc, pre, gt = flat_accuracy(preds, label_ids)
            total_eval_accuracy += acc

            precision_list.append(precision_score(gt, pre, average='micro'))
            r_s = recall_score(gt, pre, average='macro')
            recall_list.append(r_s)
            f1_s = f1_score(gt, pre, average='macro')
            f1_macro_list.append(f1_s)
            try:
                rocauc = roc_auc_score(y_true=pre, y_score=pred_prob, multi_class='ovr')
                rocauc_list.append(rocauc)
            except ValueError:
                logging.warning("Some class not shown in the mini-batch")
            wandb.log({'Accuracy': acc, "Recall Score": r_s, "F1 Macro Score": f1_s})

        avg_val_loss = total_eval_loss / len(val_dataloader)

        wandb.log({'Validation Loss': avg_val_loss})
        try:
            wandb.log({'ROC AUC Score': sum(rocauc_list) / len(rocauc_list)})
        except:
            logging.info(" ROC AUC Score cannot be calculated because of divide by zero")

    print("Training complete!")

def evalERNIE(model, test_dataloader, device):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    f1_macro_list = []
    recall_list = []
    precision_list = []
    rocauc_list = []
    y_true = []
    y_pred = []

    # Evaluate data for one epoch
    for step, (data, att_msk, label, ents, ent_msk) in enumerate(tqdm(test_dataloader)):
        data, att_msk, label, ent, ent_msk = data.to(device), att_msk.to(device), label.to(device), ents.to(
            device), ent_msk.to(device)


        with torch.no_grad():
            preds, loss = model(data, att_msk, ent, ent_msk, label)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        pred_prob = preds.softmax(dim=1).detach().cpu().numpy()

        # Move logits and labels to CPU
        preds = preds.detach().cpu().numpy()
        label_ids = label.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        acc, pre, gt = flat_accuracy(preds, label_ids)
        total_eval_accuracy += acc

        p_s = precision_score(gt, pre, average='micro')
        precision_list.append(p_s)
        r_s = recall_score(gt, pre, average='macro')
        recall_list.append(r_s)
        f1_s = f1_score(gt, pre, average='macro')
        f1_macro_list.append(f1_s)
        try:
            rocauc = roc_auc_score(y_true=pre, y_score=pred_prob, multi_class='ovr')
            rocauc_list.append(rocauc)
        except ValueError:
            logging.warning("Some class not shown in the mini-batch")
        wandb.log({'Test Accuracy': acc, "Test Recall Score": r_s, "Test F1 Macro Score": f1_s, "Test precision Score": p_s})
        y_true.append(gt)
        y_pred.append(pre)

    y_true_m = list(itertools.chain.from_iterable(y_true))
    y_pred_m = list(itertools.chain.from_iterable(y_pred))
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                       preds=y_true_m, y_true=y_pred_m,
                                                       class_names=["NA", "therapeutic", "biomarker",
                                                                    "genomic_alterations"])})

    avg_val_loss = total_eval_loss / len(test_dataloader)

    wandb.log({'Test Loss': avg_val_loss})
    try:
        wandb.log({'ROC AUC Score': sum(rocauc_list) / len(rocauc_list)})
    except:
        logging.info(" ROC AUC Score cannot be calculated because of divide by zero")
