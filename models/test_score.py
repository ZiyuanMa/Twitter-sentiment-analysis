import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
import torch
import multiprocessing
pool_num = round(multiprocessing.cpu_count()/2)
from model1 import BiLSTM, model
from fastai.text import load_learner

def predict(text):
    prob2 = m2.predict(text)[2]

    prob1 = m1.predict(text)[1]

    prob3 = torch.cat([prob1, prob2], dim=-1)

    idx1 = torch.max(prob1, 0)[1].item()
    idx2 = torch.max(prob2, 0)[1].item()
    idx3 = torch.max(prob3, 0)[1].item()%3

    return idx1, idx2, idx3



if __name__ == '__main__':

    print("start testing, this would take some time\n")
    label_to_idx = {'negative':0, 'neutral': 1, 'positive': 2}
    idx_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    label_names = ['negative', 'neutral', 'positive']
    test = pd.read_csv("../data/modified_data/testing_data.csv",header=None,encoding = "latin-1")

    raw_text = []
    with open('../data/raw_data/SemEval2017-task4-test.subtask-A.english.txt') as f:
        for line in f:
            s = line.strip().split('\t')
            raw_text.append(s)

    answer = test[0].apply(lambda x: label_to_idx[x]).tolist()

    # baseline score
    baseline = test[0].apply(lambda x: 2).tolist()

    with open('../output/baseline_output.txt','w') as f:
        for raw, out in zip(raw_text, baseline):
            f.write(raw[0]+'\t'+idx_to_label[out]+'\t'+raw[2]+'\n')

    macro_avg = classification_report(answer, baseline, target_names=label_names, output_dict=True)['macro avg']

    print('baseline score: ')
    print('\tprecision\trecall\t\tf1-score')
    print('\t%.4f\t\t%.4f\t\t%.4f' % (macro_avg['precision'], macro_avg['recall'], macro_avg['f1-score']))
    print()

    m1 = model('model1')
    m1.load_model()
    m2 = load_learner('./','model_data/model2.pkl')

    model1_out = []
    model2_out = []
    model3_out = []

    with multiprocessing.Pool(pool_num) as p:
        models_out = p.map(predict, test[1])

    model1_out = [out for out,_,_ in models_out]
    model2_out = [out for _,out,_ in models_out]
    model3_out = [out for _,_,out in models_out]

    with open('../output/model1_output.txt','w') as f:
        for raw, out in zip(raw_text, model1_out):
            f.write(raw[0]+'\t'+idx_to_label[out]+'\t'+raw[2]+'\n')

    macro_avg = classification_report(answer, model1_out, target_names=label_names, output_dict=True)['macro avg']
    print('model 1 score:')
    print('\tprecision\trecall\t\tf1-score')
    print('\t%.4f\t\t%.4f\t\t%.4f' % (macro_avg['precision'], macro_avg['recall'], macro_avg['f1-score']))
    print()

    with open('../output/model2_output.txt','w') as f:
        for raw, out in zip(raw_text, model2_out):
            f.write(raw[0]+'\t'+idx_to_label[out]+'\t'+raw[2]+'\n')

    macro_avg = classification_report(answer, model2_out, target_names=label_names, output_dict=True)['macro avg']
    print('model 2 score:')
    print('\tprecision\trecall\t\tf1-score')
    print('\t%.4f\t\t%.4f\t\t%.4f' % (macro_avg['precision'], macro_avg['recall'], macro_avg['f1-score']))
    print()

    with open('../output/model3_output.txt','w') as f:
        for raw, out in zip(raw_text, model3_out):
            f.write(raw[0]+'\t'+idx_to_label[out]+'\t'+raw[2]+'\n')

    macro_avg = classification_report(answer, model3_out, target_names=label_names, output_dict=True)['macro avg']
    print('model 3 score:')
    print('\tprecision\trecall\t\tf1-score')
    print('\t%.4f\t\t%.4f\t\t%.4f' % (macro_avg['precision'], macro_avg['recall'], macro_avg['f1-score']))
    print()


