
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import os, sys
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
import numpy as np
import pandas as pd
import pickle
torch.manual_seed(42)
torch.cuda.manual_seed(42)

word_to_ix = pickle.load(open('../data/modified_data/word_to_idx.pkl','rb'))
word_embeddings = pickle.load(open('../data/modified_data/word_embeddings.pkl','rb'))
# load dictionary and word embedding
# word_to_ix = dict()
# word_embeddings = torch.empty([658125,300],dtype=torch.float32)
# with open('data/datastories.twitter.300d.txt') as f:
#     for i, line in enumerate(f):
#         line = line.strip().split()
#         word_to_ix[line[0]] = len(word_to_ix)
#         word_vec = line[1:]
#         word_vec = [float(num) for num in word_vec]
#         word_vec = np.asarray(word_vec)
#         word_embeddings[i] = torch.from_numpy(word_vec)


punctuation = ('!', '.', ',', '?')
def clean_data(text):

    tokens = []

    for token in text.split():
        token = token.lower()
        if token[-1] in punctuation:
            for i in range(len(token)):
                if token[i] in punctuation:
                    tokens.append(token[:i])
                    tokens.append(token[i:])
                    break
        else:
            tokens.append(token)
    return tokens


# override pytorch dataset
class DealDataset(Dataset):

    def __init__(self, data_type):

        tag_to_idx = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.x_data = []
        self.y_data = []

        data = pd.read_csv("../data/modified_data/"+data_type+"ing_data.csv",header=None,encoding = "latin-1")
        for row in data.iterrows():

            pp_sent = clean_data(row[1][1])

            if len(pp_sent) > 0:

                self.x_data.append(pp_sent)
                self.y_data.append(tag_to_idx[row[1][0]])


    def __getitem__(self, index):
        
        sent_index = list(map(self.word2idx, self.x_data[index]))
        return torch.tensor(sent_index,dtype=torch.int64), self.y_data[index], len(sent_index)

    def __len__(self):
        return len(self.y_data)

    def word2idx(self, word):
        if word in word_to_ix:
            return word_to_ix[word]
        else:
            return word_to_ix['<unk>']



def pad_batch(batch):

    batch.sort(key= lambda x: x[2], reverse=True)
    sent_batch = pad_sequence([item[0] for item in batch])
    tag_batch = torch.tensor([item[1] for item in batch])
    sent_length = [item[2] for item in batch]

    return sent_batch, tag_batch, sent_length


def get_data_loader():


    train_dataset = DealDataset('train')

    train_length = round(len(train_dataset)*0.8)
    valid_length = len(train_dataset) - train_length

    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_length, valid_length])
    train_loader = DataLoader(dataset=train_dataset,batch_size=150,shuffle=True,collate_fn=pad_batch)
    valid_loader = DataLoader(dataset=valid_dataset,batch_size=150,shuffle=True,collate_fn=pad_batch)

    test_dataset = DealDataset('test')
    test_loader = DataLoader(dataset=test_dataset,batch_size=150,shuffle=True,collate_fn=pad_batch)
    return train_loader, valid_loader, test_loader

def get_word_dict():
    #word_to_ix = pickle.load(open('../data/modified_data/word_to_idx.pkl','rb'))
    return word_to_ix

def get_word_embeddings():
    #word_embeddings = pickle.load(open('../data/modified_data/word_embeddings.pkl','rb'))

    return word_embeddings