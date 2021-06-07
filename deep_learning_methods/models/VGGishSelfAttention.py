from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchvggish.vggish import VGGish

class VGGForClassificationWithAudio(nn.Module):
    def __init__(self,num_gender_features):
        super(VGGForClassificationWithAudio, self).__init__()
        if(num_gender_features == 0):
            self.useGender = False
        else:
            self.useGender = True 
        self.num_labels = 2
        self.vggish_layer = VGGish()

        #------------------------------
        input_dim = 128
        #LSTM
        print("Always Using LSTM for SelfAttention")
        self.useLSTM = True
        hidden_dim = 128
        n_layers = 1     
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        
        #Attention Layers
        self.W_s1_audio = nn.Linear(hidden_dim, 350)
        self.W_s2_audio = nn.Linear(350, 30)
        self.fc_layer = nn.Linear(31*hidden_dim, 2000)    
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(2000+num_gender_features, self.num_labels)
        
    def attention_net(self, lstm_output_audio):

        """
        Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
        encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of 
        the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully 
        connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e., 
        pos & neg.

        Arguments
        ---------

        lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
        ---------

        Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
                  attention to different parts of the input sentence.

        Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
                      attn_weight_matrix.size() = (batch_size, 30, num_seq)

        """
        attn_weight_matrix_audio = self.W_s2_audio(F.tanh(self.W_s1_audio(lstm_output_audio)))
        attn_weight_matrix_audio = attn_weight_matrix_audio.permute(0, 2, 1)
        attn_weight_matrix_audio = F.softmax(attn_weight_matrix_audio, dim=2)

        return attn_weight_matrix_audio
    
        
    def forward(self, gender_ids, wav_file, labels=None):
        audio_outputs = self.vggish_layer(wav_file)
        #print(audio_outputs.size())
        audio_outputs = audio_outputs.unsqueeze(0)
        audio_out, (ht, ct) = self.lstm_layer(torch.tensor(audio_outputs))
        hidden_audio_output = ht[-1]
        #---
        attn_weight_matrix_audio = self.attention_net(audio_out)
        hidden_matrix_audio = torch.bmm(attn_weight_matrix_audio, audio_out)
        audio_attention_output = hidden_matrix_audio.view(-1, hidden_matrix_audio.size()[1]*hidden_matrix_audio.size()[2])
        audio_pooled_output = torch.cat((audio_attention_output,hidden_audio_output),1)

        fc_out = self.fc_layer(audio_pooled_output)

        if self.useGender:
            logits = self.classifier(torch.cat((fc_out,gender_ids),1))
        else:
            logits = self.classifier(fc_out)
       
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
