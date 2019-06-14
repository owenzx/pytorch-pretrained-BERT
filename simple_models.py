import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss






def init_simple_weights(module):
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        nn.init.xavier_uniform(module.weight)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()



class SimpleSequenceClassification(nn.Module):
    def __init__(self, config, num_labels, glove_embs = None, freeze_emb=True):
        super(SimpleSequenceClassification, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.LSTM = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=config.num_lstm_layers, dropout=config.hidden_dropout_prob, bidirectional=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.MLP_hidden),
            nn.ReLU(),
            nn.Linear(config.MLP_hidden, num_labels)
        )
        self.apply(init_simple_weights)
        if glove_embs is not None:
            glove_embs = torch.tensor(glove_embs).float().cuda()
            self.word_embeddings = nn.Embedding.from_pretrained(glove_embs, freeze=freeze_emb, padding_idx=0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.dropout(embeddings)

        hiddens, _ = self.LSTM(embeddings)



        #pooled_output = torch.sum(hiddens*attention_mask.float().unsqueeze(-1), dim=-2) / torch.sum(attention_mask).float()
        pooled_output = torch.max(hiddens, dim=-2)[0]


        #pooled_output = torch.mean(hiddens, -1)
        pooled_output = self.dropout(pooled_output)


        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits




    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        pass
