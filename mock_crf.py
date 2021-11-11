# https://towardsdatascience.com/pos-tagging-using-crfs-ea430c5fb78b
# https://medium.com/analytics-vidhya/pos-tagging-using-conditional-random-fields-92077e5eaa31


# MAKE SURE TO CITE towardsdatascience, code is straight copy-pasted

# importing all the needed libraries
import pandas as pd
import nltk
import sklearn
import sklearn_crfsuite
import scipy.stats
import math, string, re

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer

#reading and storing the data
data = {}
#data['test']  = pd.read_csv('random/hi-ud-test .conllu', sep='\t')
#data['train'] = pd.read_csv('random/hi-ud-train.conllu')
# data['test'] = pd.read_csv('random/en_dev.txt', error_bad_lines=False, engine='python', sep='\t')
# data['train'] = pd.read_csv('random/en_train.txt', error_bad_lines=False, engine='python', sep='\t')
data['test'] = pd.read_csv('random/en_dev.txt', sep='\t')
data['train'] = pd.read_csv('random/en_train.txt', sep='\t')

#pd.set_option("display.max_rows", None, "display.max_columns", None)
print(data['train'], data['test'], sep = '\n\n')
#print(data['test'])

def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word': word,
        'len(word)': len(word),
        'word[:4]': word[:4],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[-4:]': word[-4:],
        'word.lower()': word.lower(),
        'word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word.lower()),
        'word.ispunctuation': (word in string.punctuation),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word': word1,
            '-1:len(word)': len(word1),
            '-1:word.lower()': word1.lower(),
            '-1:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word1.lower()),
            '-1:word[:3]': word1[:3],
            '-1:word[:2]': word1[:2],
            '-1:word[-3:]': word1[-3:],
            '-1:word[-2:]': word1[-2:],
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.ispunctuation': (word1 in string.punctuation),
        })

    else:
        features['BOS'] = True

    if i > 1:
        word2 = sent[i-2][0]
        features.update({
            '-2:word': word2,
            '-2:len(word)': len(word2),
            '-2:word.lower()': word2.lower(),
            '-2:word[:3]': word2[:3],
            '-2:word[:2]': word2[:2],
            '-2:word[-3:]': word2[-3:],
            '-2:word[-2:]': word2[-2:],
            '-2:word.isdigit()': word2.isdigit(),
            '-2:word.ispunctuation': (word2 in string.punctuation),
        })

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word': word1,
            '+1:len(word)': len(word1),
            '+1:word.lower()': word1.lower(),
            '+1:word[:3]': word1[:3],
            '+1:word[:2]': word1[:2],
            '+1:word[-3:]': word1[-3:],
            '+1:word[-2:]': word1[-2:],
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.ispunctuation': (word1 in string.punctuation),
        })

    else:
        features['EOS'] = True

    if i < len(sent) - 2:
        word2 = sent[i+2][0]
        features.update({
            '+2:word': word2,
            '+2:len(word)': len(word2),
            '+2:word.lower()': word2.lower(),
            '+2:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word2.lower()),
            '+2:word[:3]': word2[:3],
            '+2:word[:2]': word2[:2],
            '+2:word[-3:]': word2[-3:],
            '+2:word[-2:]': word2[-2:],
            '+2:word.isdigit()': word2.isdigit(),
            '+2:word.ispunctuation': (word2 in string.punctuation),
        })

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [word[1] for word in sent]

def sent2tokens(sent):
    return [word[0] for word in sent]

#formatting the data into sentences
def format_data(csv_data):
    sents = []
    for i in range(len(csv_data)):
        if math.isnan(float(csv_data.iloc[i, 0])):
            continue
        elif float(csv_data.iloc[i, 0]) == 1:
        	sents.append([[csv_data.iloc[i, 1], csv_data.iloc[i, 2]]])
        else:
            sents[-1].append([csv_data.iloc[i, 1], csv_data.iloc[i, 2]])
    for sent in sents:
        for i, word in enumerate(sent):
            if type(word[0]) != str:
                del sent[i]
    return sents

#extracting features from all the sentences
train_sents = format_data(data['train'])
test_sents = format_data(data['test'])

Xtrain = [sent2features(s) for s in train_sents]
ytrain = [sent2labels(s) for s in train_sents]

Xtest = [sent2features(s) for s in test_sents]
ytest = [sent2labels(s) for s in test_sents]

crf = sklearn_crfsuite.CRF(
    algorithm = 'lbfgs',
    c1 = 0.25,
    c2 = 0.3,
    max_iterations = 100,
    all_possible_transitions=True
)
crf.fit(Xtrain, ytrain)   

#obtaining metrics such as accuracy, etc. on the train set
labels = list(crf.classes_)
#labels.remove('X')

ypred = crf.predict(Xtrain)
print('F1 score on the train set = {}\n'.format(metrics.flat_f1_score(ytrain, ypred, average='weighted', labels=labels)))
print('Accuracy on the train set = {}\n'.format(metrics.flat_accuracy_score(ytrain, ypred)))

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print('Train set classification report: \n\n{}'.format(metrics.flat_classification_report(
    ytrain, ypred, labels=sorted_labels, digits=3
)))

#obtaining metrics such as accuracy, etc. on the test set
ypred = crf.predict(Xtest)
print('F1 score on the test set = {}\n'.format(metrics.flat_f1_score(ytest, ypred,
                      average='weighted', labels=labels)))
print('Accuracy on the test set = {}\n'.format(metrics.flat_accuracy_score(ytest, ypred)))

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print('Test set classification report: \n\n{}'.format(metrics.flat_classification_report(
    ytest, ypred, labels=sorted_labels, digits=3
)))

# need to spend a day and figure all of this out, our feature vector should not be so dense


#obtaining the most likely and the least likely transitions 
from collections import Counter

def print_transitions(transition_features):
    for (label_from, label_to), weight in transition_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top 10 likely transitions - \n")
print_transitions(Counter(crf.transition_features_).most_common(10))

print("\nTop 10 unlikely transitions - \n")
print_transitions(Counter(crf.transition_features_).most_common()[-10:])

