from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchvggish.vggish import VGGish


class VGGForClassificationWithAudio(nn.Module):
    def __init__(self,num_gender_features,numVectors,useLSTM,useLastVectors):
        super(VGGForClassificationWithAudio, self).__init__()
        if(num_gender_features == 0):
            self.useGender = False
        else:
            self.useGender = True 
            
        if useLastVectors:
            self.useLastVectors = True
        else:
            self.useLastVectors = False    
        self.numVectors = numVectors
        input_dim = 128
        if useLSTM:
            #LSTM
            print("using LSTM")
            self.useLSTM = True
            hidden_dim = 128
            n_layers = 1     
            self.lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
            self.classifier = nn.Linear(hidden_dim+num_gender_features, 2)
        else:
            self.useLSTM = False
            self.classifier = nn.Linear((numVectors * input_dim)+num_gender_features, 2)
        
        self.num_labels = 2
        self.vggish_layer = VGGish()
        self.dropout = nn.Dropout(0.1)

    def forward(self, gender_ids, wav_file, labels=None):
        audio_outputs = self.vggish_layer(wav_file)
        if self.useLSTM:
            audio_outputs = audio_outputs.unsqueeze(0)
            lstm_out, (ht, ct) = self.lstm_layer(torch.tensor(audio_outputs))
            audio_outputs = ht[-1]
            pooled_output = self.dropout(audio_outputs)
        #---
        else:
            if self.useLastVectors:               
                maxVectors = audio_outputs.size()[0] - 1 
                initial = maxVectors
                vectorRange = range(maxVectors-1,maxVectors-self.numVectors,-1)
            else:
                initial = 0
                vectorRange = range(1, self.numVectors,1)
            
            pooled_output = torch.tensor([audio_outputs[initial].tolist()]).cuda()
            
            for wavVector in vectorRange:
                audioTempOutput = torch.tensor([audio_outputs[wavVector].tolist()]).cuda()
                pooled_output = torch.cat((pooled_output,audioTempOutput),1)        
        
        
        if self.useGender:
            logits = self.classifier(torch.cat((pooled_output,gender_ids),1))
        else:
            logits = self.classifier(pooled_output)
       
        outputs = (logits,)  # add hidden states and attention if they are here

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
