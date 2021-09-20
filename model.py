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
SOS = 0
EOS = 1
class Embed:
    def __init__(self):
        self.w2id = {}
        self.w2cnt = {}
        self.id2w = {0: "SOS", 1: "EOS"}
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
    emotions = []
    with open('stc-3_emotion_train.json', encoding='utf-8') as f:
        fileJson = json.load(f)
        cntt = 0;   # train dataset size
        punctuation = r''   # punctuation filter
        for i in fileJson:
            cntt += 1
            if cntt == 3000:
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
            pairs.append([g, c, int(i[1][1])])
    # pairs = [[]]  # need to handle the data
    questions = Embed()
    answers = Embed()
    for pair in pairs:
        questions.addSentence(pair[0])
        answers.addSentence(pair[1])
    return questions, answers, pairs
input_lang, output_lang, pairs = prepareData()
class Encoder(nn.Module):
    def __init__(self, inputSize, HiddenSize):
        super(Encoder, self).__init__()
        self.hiddenSize = HiddenSize
        self.embedding = nn.Embedding(inputSize, HiddenSize)
        self.rnn = nn.GRU(HiddenSize, HiddenSize)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        result = torch.zeros(1, 1, self.hiddenSize)
        return result


class Decoder(nn.Module):
    def __init__(self, hiddenSize, outputSize, dropout_p=0.1, max_length=100):
        super(Decoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.gt = nn.Linear(hiddenSize, hiddenSize) # weight metrix for write gate
        self.gw = nn.Linear(hiddenSize, hiddenSize) # weight metrix for read gate
        self.st = nn.Sigmoid()  # write gate
        self.sw = nn.Sigmoid()  # read gate
        self.embedding = nn.Embedding(outputSize, hiddenSize)
        self.embedding2 = nn.Embedding(6, hiddenSize)   # 6 emotional vectors embedding.
        self.attn = nn.Linear(self.hiddenSize * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hiddenSize * 4, self.hiddenSize)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.GRU(hiddenSize, hiddenSize)
        self.out = nn.Linear(hiddenSize, outputSize)

    def forward(self, input, hidden, encoder_outputs, emovec, M):   # M: internal memory, emovec: index for emotional vector.
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        emovector = self.embedding2(emovec).view(1, 1, -1)
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0], emovector[0], hidden[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        # internal memory read
        ptm = self.gw(output)
        ptm = self.sw(ptm)
        hidden2 = torch.mul(M, ptm)
        output, hidden3 = self.rnn(output, hidden2)
        output = F.log_softmax(self.out(output[0]), dim=1)
        # internal memory write
        hidden4 = self.gt(hidden3)
        hidden4 = self.st(hidden4)
        hidden4 = torch.mul(hidden4, M)
        return output, hidden3, attn_weights, hidden4

    def initHidden(self):
        result = torch.zeros(1, self.hiddenSize)
        return result


ide = {}


def trainIters(encoder, decoder, n_iters, learning_rate=0.01):
    encoder_opt = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_opt = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    for itr in range(1, n_iters + 1):
        random.shuffle(pairs)
        training_pairs = [indexesFromPair(pair) for pair in pairs]
        for idx, training_pair in enumerate(training_pairs):
            inputid = training_pair[0]
            targetid = training_pair[1]
            emoid = training_pair[2]
            train(inputid, targetid, emoid, encoder, decoder, encoder_opt, decoder_opt, criterion)
teacher_forcing_ratio = 0.5
def train(inputidt, targetidt, emoid, encoder, decoder, encoder_opt, decoder_opt, criterion, max_length=100):
    encoder_hidden = encoder.initHidden()
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()
    input_length = inputidt.size()[0]
    target_length = targetidt.size()[0]
    loss = 0
    obk = []
    encoder_outputs = torch.zeros(max_length, encoder.hiddenSize)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(inputidt[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
        obk.append(input_lang.id2w[inputidt[ei].numpy()[0]])
    decoder_input = torch.LongTensor([[SOS]])
    decoder_hidden = encoder_hidden
    decoder_attentions = torch.zeros(max_length, decoder.max_length)
    decoderM = torch.ones(1, 1, decoder.hiddenSize)
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention, decoderM = decoder(decoder_input, decoder_hidden,
                                                                              encoder_outputs,
                                                                              torch.LongTensor([emoid]), decoderM)
        decoder_attentions[di] = decoder_attention.data
        loss += criterion(decoder_output, targetidt[di])
        topvalue, topindex = decoder_output.data.topk(1)
        ni = topindex[0][0].item()
        decoder_input = torch.LongTensor([[ni]])
    loss.backward()
    encoder_opt.step()
    decoder_opt.step()


def sen2id(embed, sentence):
    return [embed.w2id[word] for word in sentence.split(' ')]


def indexesFromSentence(embed, sentence):
    indexes = sen2id(embed, sentence)
    indexes.append(EOS)
    result = torch.LongTensor(indexes).view(-1, 1)
    return result


def indexesFromPair(pair):
    inputs = indexesFromSentence(input_lang, pair[0])
    targets = indexesFromSentence(output_lang, pair[1])
    return (inputs, targets, pair[2])


start_time = time.time()
hidden_size = 256
encoder = Encoder(input_lang.numW, hidden_size)
decoder = Decoder(hidden_size, output_lang.numW)
trainIters(encoder, decoder, 10)
# test case
ipt = "慕 斯 不错"
inputs = indexesFromSentence(input_lang, ipt)
input_length = inputs.size()[0]
encoder_hidden = encoder.initHidden()
encoder_outputs = torch.zeros(100, encoder.hiddenSize)
for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(inputs[ei], encoder_hidden)
    encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
decoder_input = torch.LongTensor([[SOS]])
decoder_hidden = encoder_hidden
decoderM = torch.ones(1,1,decoder.hiddenSize)
outp = []
for di in range(100):
    decoder_output, decoder_hidden, decoder_attention, decoderM = decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                                torch.LongTensor([2]), decoderM)
    topvalue, topindex = decoder_output.data.topk(1)
    ni = topindex[0][0].item()
    if ni == EOS:
        break
    else:
        outp.append(output_lang.id2w[ni])
    decoder_input = torch.LongTensor([[ni]])
print("".join(outp))
while True:
    ipt = input()   # input the first sentence for conversation.
    ipt = " ".join(jieba.cut(ipt))  # chinese sentence cut into words.
    try:
        emo = int(input())  # input wanted emotion's index for emotion embedding.
        if emo > 5 or emo < 0:
            print("false")
            continue
        inputs = indexesFromSentence(input_lang, ipt)
        input_length = inputs.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(100, encoder.hiddenSize)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(inputs[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
        decoder_input = torch.LongTensor([[SOS]])
        decoder_hidden = encoder_hidden
        outp = []
        decoderM = torch.ones(1,1,decoder.hiddenSize)
        for di in range(100):   # get output sentence.
            decoder_output, decoder_hidden, decoder_attention, decoderM = decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                                        torch.LongTensor([emo]), decoderM)
            topvalue, topindex = decoder_output.data.topk(1)
            ni = topindex[0][0].item()
            if ni == EOS:
                break
            else:
                outp.append(output_lang.id2w[ni])
            decoder_input = torch.LongTensor([[ni]])
        print("".join(outp))
    except KeyError or ValueError or IndexError:    # words are not in directory.
        print("not in directory")