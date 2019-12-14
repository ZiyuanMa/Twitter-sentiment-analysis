from twython import Twython
import json
import matplotlib.pyplot as plt
import pandas as pd
from geopy.geocoders import Nominatim
import plotly.express as px
import re
from fastai.text import load_learner
import torch
from torch import nn
import torch.nn.functional as F
import pickle
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# we need this framework to load model 1
class BiLSTM(nn.Module):
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = torch.unsqueeze(embeds, dim=1)

        # LSTM layer
        lstm_out, (hidden, cell) = self.lstm(embeds)
        hidden = torch.cat([hidden[-1,:,:],hidden[-2,:,:]],1)

        lstm_out = lstm_out.permute(1,0,2)

        # attention model
        scores = self.attention(lstm_out)
        scores = F.softmax(scores, dim=1)

        attn_applied = torch.bmm(lstm_out.permute(0,2,1), scores).squeeze(2)

        # FC layer
        label_space = self.fc(attn_applied)
        label_scores = F.log_softmax(label_space, dim=1)
        return label_scores


class TwAna:
    def __init__(self):
        with open("./tweets_analysis/twitter_credentials.json", "r") as file:
            creds = json.load(file)

        self.tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'], creds['ACCESS_TOKEN'], creds['ACCESS_SECRET'])

        self.senti_model1 = torch.load('./tweets_analysis/model_data/model1.pth',map_location=device)
        self.word_to_idx = pickle.load(open('./tweets_analysis/model_data/word_to_idx.pkl','rb'))
        self.senti_model2 = load_learner('./tweets_analysis/','model_data/model2.pkl')
        self.idx_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    def search(self, topic, num = 100):

        self.data_dict = { 'text': [], 'favorite_count': [], 'location': [], 'senti_label': []}

        if num > 15:

            for status in self.tweets.search(q= topic, count=15, lang='en', result_type='popular')['statuses']:
                self.data_dict['location'].append(status['user']['location'])
                self.data_dict['text'].append(status['text'])
                self.data_dict['favorite_count'].append(status['favorite_count'])
                label = self.predict_label(status['text'])
                self.data_dict['senti_label'].append(label)
            for status in self.tweets.search(q= topic, count=num-15, lang='en')['statuses']:
                self.data_dict['location'].append(status['user']['location'])
                self.data_dict['text'].append(status['text'])
                self.data_dict['favorite_count'].append(status['favorite_count'])
                label = self.predict_label(status['text'])
                self.data_dict['senti_label'].append(label)
        else:

            for status in self.tweets.search(q= topic, count=num, lang='en', result_type='popular')['statuses']:
                self.data_dict['location'].append(status['user']['location'])
                self.data_dict['text'].append(status['text'])
                self.data_dict['favorite_count'].append(status['favorite_count'])
                label = self.predict_label(status['text'])
                self.data_dict['senti_label'].append(label)

    def display(self):

        df = pd.DataFrame.from_dict(self.data_dict)
        return display(df)

    def plot(self):
        label_count = {'negative':0, 'positive':0, 'neutral':0}
        for cat, like_count in zip(self.data_dict['senti_label'], self.data_dict['favorite_count']):
            label_count[cat] += like_count + 1
        plt.bar(range(len(label_count)), list(label_count.values()), align='center')
        plt.xticks(range(len(label_count)), list(label_count.keys()))
        plt.show()

    def plot_map(self):

        geolocator = Nominatim(user_agent="TwAna")
        cat_color = {'negative':'red', 'positive':'green', 'neutral':'blue'}
        loca_info = {'latitude': [], 'longitude': [], 'color': [], 'size': [], 'senti_label': []}
        for user_loc, cat, like_count in zip(self.data_dict['location'], self.data_dict['senti_label'], self.data_dict['favorite_count']):
            try:
                location = geolocator.geocode(user_loc)
                if location:
                    loca_info['latitude'].append(location.latitude)
                    loca_info['longitude'].append(location.longitude)
                    loca_info['color'].append(cat_color[cat])
                    loca_info['size'].append(120+like_count*0.1)
                    loca_info['senti_label'].append(cat)
            except:
                pass
 
        gapminder = px.data.gapminder().query("year==2007")
        fig = px.scatter_geo(gapminder, lat = loca_info['latitude'], lon = loca_info['longitude'], color=loca_info['senti_label'],
                     hover_name=loca_info['senti_label'], size=loca_info['size'],)

        fig.show()
    def predict_label(self, text):
        # clean data
        text = re.sub(r'RT @\S+ ','',text)
        text = re.sub(r'@\S+','',text)
        text = re.sub(r'http\S+','',text)

        indexs = []
        for word in text.split():
            if word in self.word_to_idx:
                indexs.append(self.word_to_idx[word])
            else:
                indexs.append(self.word_to_idx['<unk>'])
            
        if len(indexs) == 0:
            return 'neutral'

        # model1 predict
        with torch.no_grad():
            inputs = torch.tensor(indexs,dtype=torch.int64)
            inputs = inputs.to(device)
            label_scores = self.senti_model1(inputs)
            prob1 = torch.squeeze(label_scores, dim=0)

        prob1 = prob1.to(device).double()

        # model2 predict
        _, _, prob2 = self.senti_model2.predict(text)
        prob2 = prob2.to(device).double()

        # get maximum prob from 2 output
        prob = torch.cat([prob1, prob2], dim=-1)
        idx = torch.max(prob, 0)[1].item()%3
        
        return self.idx_to_label[idx]


if __name__ == '__main__':

    print('Z1')