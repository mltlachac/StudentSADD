from utils import pad_sequences
import random
import time
import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel, BertPreTrainedModel
from transformers import get_linear_schedule_with_warmup
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.autograd import Variable
import warnings
import pandas as pd
from sklearn.metrics import matthews_corrcoef, f1_score, auc, recall_score, precision_score, accuracy_score

warnings.filterwarnings('ignore')

# Default Configuration
MAX_LEN = 128
learning_rate = 2e-5 
epochs = 10
batch_size = 1
useAudio = True
VGGSeconds = 3
question = "text_prompt"
modelName = "BERT"
aggregatedQuestionSplits = './wav/audio_open/'
numPaseFilters = 15
numVectors = 15
useLSTM = True

traindatafile = "./data/"+question+"/train.csv"
testdatafile  = "./data/"+question+"/dev.csv" 

# Load the dataset into a pandas dataframe.
df = pd.read_csv(traindatafile)

# Get the lists of sentences, audio features and their labels.
sentences = df.text.values
labels = df.label.values

# Generate Data structures for Audio Features, Gender and Train Identifiers. 
train_identifiers = []
gender_list = []


# Audio Feature Exploration
train_identifiers = df.id2.tolist()
id_train_wav = df[['id', 'id2']]


if 'BERT' in modelName.upper():
    from transformers import BertTokenizer
    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


if 'BERT' in modelName.upper():
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    line = 0
    # For every sentence...
    for sentence in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sentence,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
        #print(encoded_sent)
        line = line + 1
    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])

if 'BERT' in modelName.upper():

    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)

    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                              value=0, truncating="post", padding="post")

    print('\nDone.')


if 'BERT' in modelName.upper():

    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

# Convert all inputs and labels into torch tensors, the required datatype 
# for our model.
if 'BERT' in modelName.upper():
    train_inputs = torch.tensor(input_ids)
    train_masks = torch.tensor(attention_masks)

train_labels = torch.tensor(labels) # All
train_identifiers = torch.tensor(train_identifiers) #All


# Create the DataLoader for our training set.
if 'BERT' in modelName.upper():
    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_identifiers)
else:
    #train_data = TensorDataset(train_labels,train_gender,train_identifiers)
    train_data = TensorDataset(train_labels,train_identifiers)

train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Load the dataset into a pandas dataframe.
df = pd.read_csv(testdatafile)

# Report the number of sentences.
print('Number of test sentences: {:,}\n'.format(df.shape[0]))

# Create sentence and label lists
sentences = df.text.values
labels = df.label.values

# Generate Data structures for Audio Features, Gender and Train Identifiers. 
test_identifiers = df.id2.tolist()
id_test_wav = df[['id', 'id2']]
    
if 'BERT' in modelName.upper():   
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        input_ids.append(encoded_sent)

    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, 
                              dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask) 

# Convert to tensors.
if 'BERT' in modelName.upper():   
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels) #All
prediction_identifiers = torch.tensor(test_identifiers) #All

# Create the DataLoader.
if 'BERT' in modelName.upper():
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels,prediction_identifiers)
else:
    #prediction_data = TensorDataset(prediction_labels,prediction_gender,prediction_identifiers)
    prediction_data = TensorDataset(prediction_labels,prediction_identifiers)

prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

if modelName.upper() == 'BERT':
    from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel, BertPreTrainedModel
else:
    from transformers import AdamW

if modelName.upper() == 'BERT':
    from models.BERTBase import BERTBase
    model = BERTBase.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        num_gender_features = num_gender_features,
        numPaseFilters = numPaseFilters)  
if modelName.upper() == 'BERTLSTM':
    from models.BERTLSTM import BERTLSTM
    model = BERTLSTM.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        num_gender_features = num_gender_features,
        numPaseFilters = numPaseFilters) 
if modelName.upper() == 'BERTSELFATTENTION':
    from models.BERTSelfAttention import BERTSelfAttention
    model = BERTSelfAttention.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        num_gender_features = num_gender_features,
        numPaseFilters = numPaseFilters)     

if modelName.upper() == 'VGG':
    from torchvggish.vggish import VGGish
    from models.VGGish import VGGForClassificationWithAudio    
    model = VGGForClassificationWithAudio(num_gender_features = num_gender_features, numVectors = numVectors, useLSTM = useLSTM, useLastVectors = useLastVectors)

elif modelName.upper() == 'VGGISHSELFATTENTION':
    from torchvggish.vggish import VGGish
    from models.VGGishSelfAttention import VGGForClassificationWithAudio    
    model = VGGForClassificationWithAudio(num_gender_features = num_gender_features)    
    
# Tell pytorch to run this model on the GPU.
model.cuda()    

optimizer = AdamW(model.parameters(),
                  lr = learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 2e-8 # args.adam_epsilon  - default is 1e-8.
                )

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []
mcc_values = []
fone_values = []
acc_values = []

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        if 'BERT' in modelName.upper():           
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_gender =torch.tensor([1]).to(device)
            id_for_wav = id_train_wav[id_train_wav['id2'] == batch[3].cpu().numpy()[0]].id.iloc[0]
            wav_file = aggregatedQuestionSplits+str(id_for_wav)+'.wav'
        else:
            b_labels = batch[0].to(device)
            b_gender =torch.tensor([1]).to(device)
            id_for_wav = id_train_wav[id_train_wav['id2'] == batch[1].cpu().numpy()[0]].id.iloc[0]
            wav_file = aggregatedQuestionSplits+str(id_for_wav)+'.wav'

        model.zero_grad()        

        if 'BERT' in modelName.upper():           
            outputs = model(b_input_ids, 
                        b_gender,
                        wav_file,
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
        else:
            outputs = model(b_gender,wav_file,labels=b_labels)

        loss = outputs[0]

        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
    
    print("Testing:")

    # Prediction on test set

    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        if 'BERT' in modelName.upper():           
            b_input_ids, b_input_mask, b_labels, b_identifier = batch
            b_gender =torch.tensor([1]).to(device)
            id_for_wav = id_test_wav[id_test_wav['id2'] == batch[3].cpu().numpy()[0]].id.iloc[0]
            wav_file = aggregatedQuestionSplits+str(id_for_wav)+'.wav'
        else:
            b_labels = batch[0].to(device)

            b_gender =torch.tensor([1]).to(device)
            id_for_wav = id_test_wav[id_test_wav['id2'] == batch[1].cpu().numpy()[0]].id.iloc[0]
            wav_file = aggregatedQuestionSplits+str(id_for_wav)+'.wav'

        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
            if 'BERT' in modelName.upper():
                outputs = model(b_input_ids, b_gender,wav_file, token_type_ids=None, attention_mask=b_input_mask)
            else:
                outputs = model(b_gender,wav_file)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
    print('DONE.')    
    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    trueProbability = []
    falseProbability = []

    for prediction in predictions:
        falseProbability.append(float(prediction[0][0]))
        trueProbability.append(float(prediction[0][1]))
    
    f1 = f1_score(flat_true_labels, flat_predictions)
    recall = recall_score(flat_true_labels, flat_predictions)
    precision = precision_score(flat_true_labels, flat_predictions)
    acc = accuracy_score(flat_true_labels, flat_predictions)


