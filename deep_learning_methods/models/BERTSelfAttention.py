from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel, BertPreTrainedModel
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch.autograd import Variable
from torch.nn import functional as F

class BERTSelfAttention(BertPreTrainedModel):
    def __init__(self, config,num_gender_features,numPaseFilters):
        super().__init__(config)
        if(num_gender_features == 0):
            self.useGender = False
        else:
            self.useGender = True  
        self.num_labels = self.config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)    
        embedding_dim = config.hidden_size
        self.bert_lstm_hidden_size = 128
        self.lstm_bert = nn.LSTM(embedding_dim, self.bert_lstm_hidden_size, 1,batch_first=True)
        
        #Self Attention Layer
        self.W_s1_bert = nn.Linear(self.bert_lstm_hidden_size, 350)
        self.W_s2_bert = nn.Linear(350, 30)
        self.fc_layer = nn.Linear(30*self.bert_lstm_hidden_size+num_gender_features, 2000)
        self.classifier = nn.Linear(2000, self.num_labels)
        self.init_weights()
    def attention_net_bert(self, lstm_output):
        attn_weight_matrix_bert = self.W_s2_bert(F.tanh(self.W_s1_bert(lstm_output)))
        attn_weight_matrix_bert = attn_weight_matrix_bert.permute(0, 2, 1)
        attn_weight_matrix_bert = F.softmax(attn_weight_matrix_bert, dim=2)
        return attn_weight_matrix_bert
    
    def forward(self, input_ids, gender_ids, wav_file, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, fs=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        #last_hidden_state, pooled_output = outputs
        pooled_output = outputs.pooler_output
        last_hidden_state = outputs.last_hidden_state 
        lstm_input = last_hidden_state
        lstm_out_bert, (ht, ct) = self.lstm_bert(torch.tensor(lstm_input))
        bert_lstm_hidden_output = ht[-1]  
        
        attn_weight_matrix_bert = self.attention_net_bert(lstm_out_bert)
        hidden_matrix_bert = torch.bmm(attn_weight_matrix_bert, lstm_out_bert)
        bert_attention_output = hidden_matrix_bert.view(-1, hidden_matrix_bert.size()[1]*hidden_matrix_bert.size()[2])
        if self.useGender:
            logits = self.classifier(self.fc_layer(torch.cat((bert_attention_output,gender_ids),1)))
        else:
            logits = self.classifier(self.fc_layer(bert_attention_output))
       
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

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