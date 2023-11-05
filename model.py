import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from knowledge_bert import BertModel, BertForSequenceClassification, BertTokenizer, BertAdam, BertConfig
from configs.config import DefaultConfig as cfg

class MultiBert(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.num_labels = self.cfg.NUM_LABELS
        self.bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(p=self.cfg.DROP_OUT_PROB)
        self.hidden_dim = 3 * 768
        self.classifier = nn.Linear(self.hidden_dim, self.num_labels)

    def forward(self, input_ids, attention_mask, labels):
        output1 = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=None, output_hidden_states=True)
        pooled_output = torch.cat(tuple([output1[1][i] for i in [-3, -2, -1]]), dim=-1)
        pooled_output = pooled_output[:, 0, :]    # BSZ  3 * 768
        output = self.dropout(pooled_output)
        output = self.classifier(output)          # BSZ  num_labels

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(output, labels)
        return output, loss

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True



class ERNIEModel(nn.Module):
    def __init__(self, config=cfg):
        super().__init__()
        self.cfg = config

        # I change the num_labels in modelling.py directly line 1068
        model,_ = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.transformer = model

    def forward(self, x, att_msk, ents, ent_mask, label):
        pooled_output = self.transformer(input_ids=x, attention_mask=att_msk, input_ent=ents, ent_mask=ent_mask)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(pooled_output, label)
        return pooled_output, loss

class KBert(nn.Module):
    def __init__(self, args, model):
        super(KBert, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.use_vm = False if args.no_vm else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits

