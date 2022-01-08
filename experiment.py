from nltk import word_tokenize
from nltk import pos_tag
from sklearn.model_selection import train_test_split
import pycrfsuite, json

class Vocab(object):
    def __init__(self):
        self.vocab = {'<UNK>': 0}
        self.mixvocab = {}
        self.idx = len(self.vocab)

    def word2index(self, tokens, train=False):
        res = []
        for token in tokens:
            if token.lower() in self.vocab:
                res.append(self.vocab[token.lower()])
            else:
                if train:
                    self.vocab[token.lower()] = self.idx
                    if label(token) == 'complex':
                        self.mixvocab[token.lower()] = get_mixcase(token)
                    res.append(self.idx)
                    self.idx += 1
                else:
                    res.append(0)
        return res

def get_mixcase(token):
    res = ''
    for i in token:
        res += 'U' if i.isupper() else 'L'
    return res

def check_complex(token):
    a, b = 0, 0
    for i in token:
        if i.islower() and a == 0:
            a = 1
        if i.isupper() and b == 0:
            b = 1
        if a == b == 1:
            return True
    return False

def check_capitalize(token):
    return True if token[0].isupper() and token[1:].islower() else False

def preprocess(sentence):
    punctuation = '''!()-[]{};:'"\<>/@#$%^&*_~'''
    for sign in punctuation:
        if sign in sentence:
            sentence = sentence.replace(sign, ' ')
    return sentence

def label(token):
    if token.islower():
        return 'lower'
    elif token.isupper():
        return 'upper'
    elif check_capitalize(token):
        return 'capitalize'
    elif check_complex(token):
        return 'complex'
    else:
        return 'blank'

def extract_labels(tokens):
    res = []
    for token in tokens:
        token = token[0]
        res.append(label(token))
    return res

def extract_features(tokens):
    line = []
    features = [line.copy() for _ in range(len(tokens))]
    for i in range(0, len(tokens)):
        feature = features[i]
        '''
        feature = [word.lower, little, upper, digit, postag]
        '''
        feature.extend([
            tokens[i][0].lower(),
            tokens[i][0][-3:],
            tokens[i][0][-2:],
            str(tokens[i][0].islower()),
            str(check_capitalize(tokens[i][0])),
            str(tokens[i][0].isupper()),
            str(tokens[i][0].isdigit()),
            tokens[i][1]
        ])
        if i > 0:
            feature.extend([
                tokens[i - 1][0].lower(),
                str(tokens[i - 1][0].islower()),
                str(check_capitalize(tokens[i - 1][0])),
                str(tokens[i - 1][0].isupper()),
                str(tokens[i - 1][0].isdigit()),
                tokens[i - 1][1]
            ])
        else:
            feature.append('BOS')
        if i > 1:
            feature.extend([
                tokens[i - 2][0].lower(),
                str(tokens[i - 2][0].islower()),
                str(check_capitalize(tokens[i - 2][0])),
                str(tokens[i - 2][0].isupper()),
                str(tokens[i - 2][0].isdigit()),
                tokens[i - 2][1]
            ])
        if i > 2:
            feature.extend([
                tokens[i - 3][0].lower(),
                str(tokens[i - 3][0].islower()),
                str(check_capitalize(tokens[i - 1][0])),
                str(tokens[i - 3][0].isupper()),
                str(tokens[i - 3][0].isdigit()),
                tokens[i - 3][1]
            ])
        if i < len(tokens)-3:
            feature.extend([
                tokens[i + 3][0].lower(),
                str(tokens[i + 3][0].islower()),
                str(check_capitalize(tokens[i + 3][0])),
                str(tokens[i + 3][0].isupper()),
                str(tokens[i + 3][0].isdigit()),
                tokens[i + 3][1]
            ])
        if i < len(tokens)-2:
            feature.extend([
                tokens[i + 2][0].lower(),
                str(tokens[i + 2][0].islower()),
                str(check_capitalize(tokens[i + 1][0])),
                str(tokens[i + 2][0].isupper()),
                str(tokens[i + 2][0].isdigit()),
                tokens[i + 2][1]
            ])
        if i < len(tokens)-1:
            feature.extend([
                tokens[i + 1][0].lower(),
                str(tokens[i + 1][0].islower()),
                str(check_capitalize(tokens[i + 1][0])),
                str(tokens[i + 1][0].isupper()),
                str(tokens[i + 1][0].isdigit()),
                tokens[i + 1][1]
            ])
        else:
            feature.append('EOS')

    return features

def datacollection():
    with open('data.v1.split/normal.training.txt', 'r') as f:
        lines = f.readlines()
        X = []
        y = []
        i = 0
        v = Vocab()
        for line in lines:
            if i == 20000:
                break
            line = preprocess(line)
            word_list = word_tokenize(line)
            v.word2index(word_list, train=True)
            tokens = pos_tag(word_list)
            y.append(extract_labels(tokens))
            X.append(extract_features(tokens))
            i+=1
        #print(len(X[0]), len(y[0]))
        with open('X.json', 'w') as f:
            json.dump(X, f)
        with open('y.json', 'w') as f:
            json.dump(y, f)
        with open('dic.json', 'w') as f:
            json.dump(v.vocab, f)
        return v, X, y

v, X, y = datacollection()
#with open('X.json', 'r') as f:
#    X = json.load(f)
#with open('y.json', 'r') as f:
#    y = json.load(f)

print('data collection done')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
for x in X_test:
    for i in x:
        print(i[0], end=' ')
    print('\n')
trainer = pycrfsuite.Trainer()
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)
    #print(trainer.get_params())
    trainer.set_params({'c1': 0.001, 'c2': 1, 'max_linesearch': 100})
trainer.train('crf.model')
tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

def transform(label, lower, v):
    if lower in v.mixvocab:
        format = v.mixvocab[lower]
        res = ''
        for f, l in zip(format, lower):
            res += l.upper() if f == 'U' else l
        return res

    if label == 'upper':
        return lower.upper()
    elif label == 'capitalize':
        return lower[0].upper()+lower[1:]
    else:
        return lower

def getTrueCaser(X_test, y_test, y_pred, v):
    total_correct = 0
    total = 0
    dic = {i:[0, 0] for i in ['complex', 'lower', 'upper', 'capitalize', 'complex', 'blank']}
    for X, real, pred in zip(X_test, y_test, y_pred):
        true_case = []
        real_case = []
        for x, wr, wp in zip(X, real, pred):
            lower = x[0]
            pred_word = transform(wp, lower, v)
            real_word = transform(wr, lower, v)
            true_case.append(pred_word)
            real_case.append(real_word)
            if true_case[-1]==real_case[-1]:
                total_correct += 1
                dic[wr][0] += 1
            total += 1
            dic[wr][1] += 1
        if 'complex' in real:
            print(' '.join(real_case))
            print('<><><><><><><><><><><><><><><>')
        if true_case != real_case:
            print(' '.join(true_case))
            print(' '.join(real_case))
            print('------------------------------')
    print('Total: ', total_correct/total)
    for i in dic:
        k = dic[i][1]
        if k != 0:
            print(f'{i}: ', dic[i][0]/dic[i][1])
        else:
            print(f'{i}: pass')

getTrueCaser(X_test, y_test, y_pred, v)