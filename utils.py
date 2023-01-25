import numpy as np
import os
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix
import wandb
from configs.config import cfg
import logging
import itertools
import torch.nn.functional as F


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

'''
def plot_confusion_matrix(y_true=None, y_pred=None, labels=None, true_labels=None, pred_labels=None, normalize=False):
    """
    Computes the confusion matrix to evaluate the accuracy of a classification.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    classes = np.asarray(labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0

    if true_labels is None:
        true_classes = classes
    else:
        true_label_indexes = np.in1d(classes, true_labels)
        true_classes = classes[true_label_indexes]
        cm = cm[true_label_indexes]

    if pred_labels is None:
        pred_classes = classes
    else:
        pred_label_indexes = np.in1d(classes, pred_labels)
        pred_classes = classes[pred_label_indexes]
        cm = cm[:, pred_label_indexes]

    data = []
    count = 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if labels is not None and (isinstance(pred_classes[i], int)
                                   or isinstance(pred_classes[0], np.integer)):
            pred_dict = labels[pred_classes[i]]
            true_dict = labels[true_classes[j]]
        else:
            pred_dict = pred_classes[i]
            true_dict = true_classes[j]
        data.append([pred_dict, true_dict, cm[i, j]])
        count += 1
    wandb.log({"confusion_matrix": wandb.Table(
        columns=['Predicted', 'Actual', 'Count'],
        data=data)})
    return
'''

def trainigprocessBERT(epochs, model, optimizer, scheduler, train_dataloader, validation_dataloader, device):
    # training_stats = []
    for epoch_i in range(0, epochs):

        logging.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        logging.info('Training...')
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Reset the total loss for this epoch.
        total_train_loss = 0
        running_loss = 0

        model.train()

        # For each batch of training data...
        for step, (seq, msk, labels) in enumerate(tqdm(train_dataloader)):
            b_input_ids = seq.to(device)
            b_input_mask = msk.to(device)
            b_labels = labels.to(device)

            model.zero_grad()

            # output shape BSZ * num_classes
            output, loss = model(input_ids=b_input_ids,attention_mask=b_input_mask,labels=b_labels)

            total_train_loss += loss.item() * output.size(0)
            running_loss += loss.item()

            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

            if step % cfg.print_every == 0:
                wandb.log({'batch_loss':  running_loss/cfg.print_every})
                running_loss = 0

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        wandb.log({'training_loss': avg_train_loss/len(train_dataloader)})
        save_model(model, optimizer, epoch_i + 1, "BERT")

        # ========================================
        #               Validation
        # ========================================
        print("Running Validation...")
        logging.info('Validation...')

        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        f1_macro_list = []
        recall_list = []
        precision_list = []
        rocauc_list = []

        # Evaluate data for one epoch
        for (seq, msk, labels) in tqdm(validation_dataloader):
            b_input_ids = seq.to(device)
            b_input_mask = msk.to(device)
            b_labels = labels.to(device)

            with torch.no_grad():
                preds, loss = model(input_ids=b_input_ids,attention_mask=b_input_mask,labels=b_labels)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            pred_prob = preds.softmax(dim=1).detach().cpu().numpy()

            # Move logits and labels to CPU
            preds = preds.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
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
                pass
                # logging.warning("Some class not shown in the mini-batch")
            wandb.log({'Accuracy': acc, "Recall Score": r_s, "F1 Macro Score": f1_s})



        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        logging.info("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        # logging.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
        #
        # logging.info("  F1 Score: {0:.4f}".format(sum(f1_macro_list)/len(f1_macro_list)))
        # logging.info("  Precision Score: {0:.4f}".format(sum(precision_list)/len(precision_list)))
        # logging.info("  Recall Score: {0:.4f}".format(sum(recall_list)/len(recall_list)))
        # logging.info("  ROC AUC Score: {0:.4f}".format(sum(rocauc_list)/len(rocauc_list)))

    #logging.info("Training complete!")

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

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        pred_prob = preds.softmax(dim=1).detach().cpu().numpy()

        # Move logits and labels to CPU
        preds = preds.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
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
            pass
            # logging.warning("Some class not shown in the mini-batch")
        wandb.log({'Test Accuracy': acc, "Test Recall Score": r_s, "Test F1 Macro Score": f1_s})
        y_true.append(gt)
        y_pred.append(pre)

    y_true_m = list(itertools.chain.from_iterable(y_true))
    y_pred_m = list(itertools.chain.from_iterable(y_pred))
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                       preds=y_true_m, y_true=y_pred_m,
                                                       class_names=["NA", "therapeutic", "biomarker", "genomic_alterations"])})
    #plot_confusion_matrix(y_true_m, y_pred_m, labels=["NA", "therapeutic", "biomarker", "genomic_alterations"])

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    logging.info("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(test_dataloader)

    logging.info("  Test Loss: {0:.2f}".format(avg_val_loss))

    #logging.info("  F1 Score: {0:.4f}".format(sum(f1_macro_list) / len(f1_macro_list)))
    #logging.info("  Precision Score: {0:.4f}".format(sum(precision_list) / len(precision_list)))
    #logging.info("  Recall Score: {0:.4f}".format(sum(recall_list) / len(recall_list)))
    try:
        # logging.info("  ROC AUC Score: {0:.4f}".format(sum(rocauc_list) / len(rocauc_list)))
        wandb.log({'ROC AUC Score': sum(rocauc_list) / len(rocauc_list)})
    except:
        logging.info(" ROC AUC Score cannot be calculated because of divide by zero")

def trainigprocessERNIE(epochs, model, optimizer, scheduler, train_dataloader, val_dataloader, device):
    training_stats = []
    for epoch_i in range(0, epochs):

        logging.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        logging.info('Training...')
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Reset the total loss for this epoch.
        total_train_loss = 0
        running_loss = 0

        model.train()

        # For each batch of training data...
        for step, (data, att_msk, label, ents, ent_msk) in enumerate(tqdm(train_dataloader)):
            data, att_msk, label, ent, ent_msk = data.to(device), att_msk.to(device), label.to(device), ents.to(device), ent_msk.to(device)
            #             print(f'{data.shape=}')
            #             print(f'{ent.shape=}')
            #             print(f'{ent_msk.shape=}')

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

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        wandb.log({'training_loss': avg_train_loss / len(train_dataloader)})

        save_model(model, optimizer, epoch_i + 1, "ERNIE")
        # ========================================
        #               Validation
        # ========================================
        print("Running Validation...")
        logging.info('Validation...')

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

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
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
                pass
                # logging.warning("Some class not shown in the mini-batch")
            wandb.log({'Accuracy': acc, "Recall Score": r_s, "F1 Macro Score": f1_s})

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        logging.info("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_dataloader)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        # logging.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
        # logging.info("  F1 Score: {0:.4f}".format(sum(f1_macro_list) / len(f1_macro_list)))
        # logging.info("  Precision Score: {0:.4f}".format(sum(precision_list) / len(precision_list)))
        # logging.info("  Recall Score: {0:.4f}".format(sum(recall_list) / len(recall_list)))
        wandb.log({'Validation Loss': avg_val_loss})
        try:
            #logging.info("  ROC AUC Score: {0:.4f}".format(sum(rocauc_list) / len(rocauc_list)))
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
            pass
            # logging.warning("Some class not shown in the mini-batch")
        wandb.log({'Test Accuracy': acc, "Test Recall Score": r_s, "Test F1 Macro Score": f1_s, "Test precision Score": p_s})
        y_true.append(gt)
        y_pred.append(pre)

    y_true_m = list(itertools.chain.from_iterable(y_true))
    y_pred_m = list(itertools.chain.from_iterable(y_pred))
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                       preds=y_true_m, y_true=y_pred_m,
                                                       class_names=["NA", "therapeutic", "biomarker",
                                                                    "genomic_alterations"])})
    # plot_confusion_matrix(y_true_m, y_pred_m, labels=["NA", "therapeutic", "biomarker", "genomic_alterations"])

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    logging.info("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(test_dataloader)

    logging.info("  Test Loss: {0:.2f}".format(avg_val_loss))
    wandb.log({'Test Loss': avg_val_loss})
    #logging.info("  F1 Score: {0:.4f}".format(sum(f1_macro_list) / len(f1_macro_list)))
    #logging.info("  Precision Score: {0:.4f}".format(sum(precision_list) / len(precision_list)))
    #logging.info("  Recall Score: {0:.4f}".format(sum(recall_list) / len(recall_list)))
    try:
        # logging.info("  ROC AUC Score: {0:.4f}".format(sum(rocauc_list) / len(rocauc_list)))
        wandb.log({'ROC AUC Score': sum(rocauc_list) / len(rocauc_list)})
    except:
        logging.info(" ROC AUC Score cannot be calculated because of divide by zero")

if __name__ == '__main__':
    pass
    # from sklearn.datasets import load_iris
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import roc_auc_score
    #
    # X, y = load_iris(return_X_y=True)
    # clf = LogisticRegression(solver="liblinear").fit(X, y)
    # proob =  clf.predict_proba(X)
    # roc_auc_score(y, clf.predict_proba(X), multi_class='ovr')