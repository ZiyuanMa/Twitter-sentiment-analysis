
# Bi-LSTM with attention

import os, sys
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import numpy as np
import re
from load_data import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.manual_seed(1261)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1261)

class BiLSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim, hidden_dim):

        super(BiLSTM, self).__init__()
        self.device = device
        word_embeddings = get_word_embeddings()
        self.word_embeddings = nn.Embedding.from_pretrained(word_embeddings)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.25)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 3),
        ) 
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim*2, 1),
            nn.Tanh()
        )
    def forward(self, sentence, sent_len=None):
        embeds = self.word_embeddings(sentence)
        if sent_len:
            embeds = pack_padded_sequence(embeds, sent_len)
        else:
            embeds = torch.unsqueeze(embeds, dim=1)

        # LSTM layer
        lstm_out, (hidden, cell) = self.lstm(embeds)
        hidden = torch.cat([hidden[-1,:,:],hidden[-2,:,:]],1)

        if sent_len:
            lstm_out, input_sizes = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out = lstm_out.permute(1,0,2)

        # attention model
        scores = self.attention(lstm_out)

        if sent_len:
            mask = torch.arange(input_sizes[0]).expand(len(input_sizes), input_sizes[0]) < input_sizes.unsqueeze(1)
            mask = mask.unsqueeze(2).to(self.device)
            scores = scores.masked_fill(mask == False, -np.inf)
        scores = F.softmax(scores, dim=1)
        attn_applied = torch.bmm(lstm_out.permute(0,2,1), scores).squeeze(2)

        # FC layer
        tag_space = self.fc(attn_applied)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class model:

    def __init__(self, model_name, epochs=5, embedding_dim=300, hidden_dim=150):
        

        self.device = device
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.word_to_idx = get_word_dict()
        self.model = BiLSTM(len(self.word_to_idx),self.embedding_dim, self.hidden_dim)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.99))


    def train(self):
        self.train_data, self.vaild_data, self.test_data = get_data_loader()
        self.idx_to_tag = {0: 'negative', 1:'neutral', 2:'positive'}

        loss_function = nn.NLLLoss()

        loss = float("inf")

        # start train
        for epoch in range(self.epochs):
            self.model.train()
            sum_loss = 0
            for sentence, tags, sent_len in tqdm(self.train_data):
                self.model.zero_grad()
                sentence, tags = sentence.to(self.device), tags.to(self.device)
                tag_scores = self.model(sentence, sent_len)
                loss = loss_function(tag_scores, tags)
                sum_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            print('total loss: %.2f' %sum_loss)
            self.test(self.train_data, 'train_data')
            self.test(self.vaild_data, 'vaild_data')

        # save model
        savefile = './model_data/' + self.model_name + '.pth'
        print("saving model file: {}".format(savefile), file=sys.stderr)

        torch.save(self.model, savefile)
        self.test(self.test_data, 'test_data')
        
    def load_model(self):
        # saved_model = torch.load('models/model_data/'+self.model_name + '.pth',map_location=self.device)
        # self.model.load_state_dict(saved_model['model_state_dict'])
        # self.model.load('./')
        self.model = torch.load('./model_data/model1.pth',map_location=self.device)
        self.word_to_idx = get_word_dict()
        self.idx_to_tag = {0: 'negative', 1:'neutral', 2:'positive'}
        self.model.eval()
   
    def predict(self, text):

        # word to index
        indexs = []
        for word in text.split():
            if word in self.word_to_idx:
                indexs.append(self.word_to_idx[word])
            else:
                indexs.append(self.word_to_idx['<unk>'])
            
        if len(indexs) == 0:
            return 'neutral', torch.tensor([0,1,0])
        # predict
        with torch.no_grad():
            inputs = torch.tensor(indexs,dtype=torch.int64)
            inputs = inputs.to(self.device)
            tag_scores = self.model(inputs)
            result = self.idx_to_tag[int(tag_scores.argmax(dim=1))]
            tag_scores = torch.squeeze(tag_scores, dim=0)
        return result, torch.exp(tag_scores)
    
    def test(self, test_data, data_name):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in test_data:
                sent, labels, sent_len = data
                sent, labels = sent.to(self.device), labels.to(self.device)
                outputs = self.model(sent, sent_len)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of ' + data_name+': %.4f' % (correct / total))


if __name__ == '__main__':
    #train model1

    m = model('model1')

    m.train()








