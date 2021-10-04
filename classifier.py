from __future__ import unicode_literals, print_function, division
from io import open
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import json
import time
import jieba
PAD = 0
EOS = 1
class Embed:
    def __init__(self):
        self.w2id = {}
        self.w2cnt = {}
        self.id2w = {0: "PAD", 1: "EOS"}
        self.numW = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.w2id:
            self.w2id[word] = self.numW
            self.w2cnt[word] = 1
            self.id2w[self.numW] = word
            self.numW += 1

def prepareData():
    # lines = open('stc-3_emotion_train.json').read().strip().split('')  # need to handle the data
    lines = []
    pairs = []
    with open('stc-3_emotion_train.json', encoding='utf-8') as f:
        fileJson = json.load(f)
        cntt = 0;   # train dataset size
        punctuation = r''   # punctuation filter
        for i in fileJson:
            cntt += 1
            if cntt == 300:
                break
            lines.append(i[0][0].strip())
            lines.append(i[1][0].strip())
            f = i[0][0].strip()
            k = f.split(' ')
            k = [x for x in k if x != '' and (not x in punctuation)]
            g = ' '.join(k)
            w = i[1][0].strip()
            o = w.split(' ')
            o = [x for x in o if x != '' and (not x in punctuation)]
            c = ' '.join(o)
            pairs.append([g, int(i[0][1])])
            pairs.append([c, int(i[1][1])])
    # pairs = [[]]  # need to handle the data
    sentences = Embed()
    for pair in pairs:
        sentences.addSentence(pair[0])
    return sentences, pairs

sentenceEmbed, pairs = prepareData()

def transform(embed, sentence, max_len = 100):
    idVector = [embed.w2id[word] for word in sentence.split(' ')]
    idVector.append(EOS)
    if max_len > len(idVector):
        idVector = idVector + [PAD] * (max_len-len(idVector))
    elif max_len < len(idVector):
        idVector = idVector[:max_len]
    return idVector

hiddenSize = 100
numLayers = 2
class Classifier(nn.Module):
    def __init__(self, inputSize, vectorSize, hiddenSize, numLayers):
        super(Classifier, self).__init__()
        self.embedding = nn.Embedding(inputSize,vectorSize)
        self.lstm = torch.nn.LSTM(input_size=vectorSize, hidden_size=hiddenSize, num_layers=numLayers, bidirectional=True, batch_first=True, dropout=0.25)
        self.fc = nn.Linear(4*hiddenSize,6)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        outputs,_ = self.lstm(embeddings)
        output = torch.cat((outputs[:,0],outputs[:,-1]),-1)
        output = self.fc(output)

        return output

lr, num_epoch = 0.001,50
classifier = Classifier(sentenceEmbed.numW,100,hiddenSize, numLayers)
optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
from sklearn.model_selection import train_test_split
data_input1 = []
data_input2 = []
for pair in pairs:
    data_input1.append(transform(sentenceEmbed,pair[0]))
    data_input2.append(pair[1])
x_train, x_test, y_train, y_test = train_test_split(data_input1,data_input2,test_size=0.2,random_state=None)

import torch.utils.data as Data
torch_dataset = Data.TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
loader = Data.DataLoader(dataset=torch_dataset, batch_size=30, shuffle=True)
for iterate in range(num_epoch):
    for index, label in enumerate(loader):
        optimizer.zero_grad()
        output = classifier(label[0])
        loss = criterion(output,label[1])
        loss.backward()
        optimizer.step()
    print(loss)

test_result = classifier(torch.LongTensor(x_test))
_, ind = torch.topk(test_result,1,dim=1)
correct = 0
for i in range(len(y_test)):
    if ind[i][0] == y_test[i]:
        correct += 1
print(correct/len(y_test))