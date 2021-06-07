from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel, BertPreTrainedModel
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch.autograd import Variable
from torch.nn import functional as F

class BERTBase(BertPreTrainedModel):
    def __init__(self, config,num_gender_features,numPaseFilters):
        super().__init__(config)
        if(num_gender_features == 0):
            self.useGender = False
        else:
            self.useGender = True  
        self.num_labels = self.config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)    
        self.classifier = nn.Linear(config.hidden_size+num_gender_features, self.config.num_labels)
        self.init_weights()
    
    def forward(self, input_ids, gender_ids, wav_file, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, fs=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs.pooler_output
        last_hidden_state = outputs.last_hidden_state 

        if self.useGender:
            logits = self.classifier(torch.cat((pooled_output,gender_ids),1))
        else:
            logits = self.classifier(pooled_output)
       
        outputs = (logits,) + outputs[2:]  

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)  