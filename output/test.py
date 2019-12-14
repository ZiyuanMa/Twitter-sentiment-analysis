# just run this file with "python3 test.py" to test score of baseline, model1, model2 and model3
# notice that this needs sklearn package 

from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    label_to_idx = {'negative':0, 'neutral': 1, 'positive': 2}
    label_names = ['negative', 'neutral', 'positive']


    answer = []
    with open('./SemEval2017-task4-test.subtask-A.english.txt') as f:
        for line in f:
            s = line.strip().split('\t')
            answer.append(label_to_idx[s[1]])

    baseline = []
    with open('./baseline_output.txt') as f:
        for line in f:
            s = line.strip().split('\t')
            baseline.append(label_to_idx[s[1]])

    macro_avg = classification_report(answer, baseline, target_names=label_names, output_dict=True)['macro avg']

    print('baseline score: ')
    print('\tprecision\trecall\t\tf1-score')
    print('\t%.4f\t\t%.4f\t\t%.4f' % (macro_avg['precision'], macro_avg['recall'], macro_avg['f1-score']))
    print()

    model1 = []
    with open('./model1_output.txt') as f:
        for line in f:
            s = line.strip().split('\t')
            model1.append(label_to_idx[s[1]])

    macro_avg = classification_report(answer, model1, target_names=label_names, output_dict=True)['macro avg']

    print('model1 score: ')
    print('\tprecision\trecall\t\tf1-score')
    print('\t%.4f\t\t%.4f\t\t%.4f' % (macro_avg['precision'], macro_avg['recall'], macro_avg['f1-score']))
    print()

    model2 = []
    with open('./model2_output.txt') as f:
        for line in f:
            s = line.strip().split('\t')
            model2.append(label_to_idx[s[1]])

    macro_avg = classification_report(answer, model2, target_names=label_names, output_dict=True)['macro avg']

    print('model2 score: ')
    print('\tprecision\trecall\t\tf1-score')
    print('\t%.4f\t\t%.4f\t\t%.4f' % (macro_avg['precision'], macro_avg['recall'], macro_avg['f1-score']))
    print()

    model3 = []
    with open('./model3_output.txt') as f:
        for line in f:
            s = line.strip().split('\t')
            model3.append(label_to_idx[s[1]])

    macro_avg = classification_report(answer, model3, target_names=label_names, output_dict=True)['macro avg']

    print('model3 score: ')
    print('\tprecision\trecall\t\tf1-score')
    print('\t%.4f\t\t%.4f\t\t%.4f' % (macro_avg['precision'], macro_avg['recall'], macro_avg['f1-score']))
    print()