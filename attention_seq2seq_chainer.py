
# coding: utf-8

# In[ ]:

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


# In[ ]:

src_vocab = {}
EOS = '<eos>'
NIL = '<nil>'
entity_vocab = {}
id2wb = {}


# In[ ]:

import pickle
def read_data_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

src_lines = read_data_from_pickle('./data/seq2seq_wiki/src/src_list_1.pickle').decode('utf8').split('\n')
dest_lines = read_data_from_pickle('./data/seq2seq_wiki/dest_all/replace_entity_list_1.pickle').decode('utf8').split('\n')
entity_lines = read_data_from_pickle('./data/seq2seq_wiki/dest_entity/entity_list_1.pickle').decode('utf8').split('\n')


# In[ ]:

len(dest_lines)


# In[ ]:

len(src_lines)


# In[ ]:

len(entity_lines)


# In[ ]:

for line in src_lines:
    lt = line.split()
    for w in lt:
        if w not in src_vocab:
            src_vocab[w] = len(src_vocab)


# In[ ]:

src_vocab[EOS] = len(src_vocab)
pv = len(src_vocab)


# In[ ]:

# entity_lines = open('./output.txt').read().split('\n')


# In[ ]:

for line in entity_lines:
    lt = line.split()
    for w in lt:
        if w not in entity_vocab:
            vocab_size = len(entity_vocab)
            entity_vocab[w] = vocab_size
            id2wb[vocab_size] = w
#     print(len(entity_vocab))


# In[ ]:

print(len(entity_vocab))
vocab_size = len(entity_vocab)
entity_vocab[EOS] = vocab_size
id2wb[vocab_size] = EOS

vocab_size = len(entity_vocab)
entity_vocab[NIL] = vocab_size
id2wb[vocab_size] = NIL

ev = len(entity_vocab)


# In[ ]:

def mk_ct(gh, ht):
    alp = []
    s = 0.0
    for i in range(len(gh)):
        s += np.exp(ht.dot(gh[i]))
    ct = np.zeros(100)
    for i in range(len(gh)):
        alpi = np.exp(ht.dot(gh[i]))/s
        ct += alpi * gh[i]
    ct = Variable(np.array([ct]).astype(np.float32))
    return ct


# In[ ]:

class ATT(chainer.Chain):
    def __init__(self, pv, ev, k):
        super(ATT, self).__init__(
            embedx = L.EmbedID(pv, k),
            embedy = L.EmbedID(ev, k),
            H = L.LSTM(k, k),
            Wc1 = L.Linear(k, k),
            Wc2 = L.Linear(k, k),
            W = L.Linear(k, ev),
        )
        
    def __call__(self, pline, eline):
        gh = []
        for i in range(len(pline)):
            wid = src_vocab[pline[i]]
            x_k = self.embedx(Variable(np.array([wid], dtype=np.int32)))
            h = self.H(x_k)
            gh.append(np.copy(h.data[0]))
            
        x_k = self.embedx(Variable(np.array([src_vocab[EOS]], dtype=np.int32)))
        tx = Variable(np.array([entity_vocab[eline[0]]], dtype=np.int32))
        h = self.H(x_k)
        ct = mk_ct(gh, h.data[0])
        h2 = F.tanh(self.Wc1(ct) + self.Wc2(h))
        accum_loss = F.softmax_cross_entropy(self.W(h2), tx)
        
        for i in range(len(eline)):
            wid = entity_vocab[eline[i]]
            x_k = self.embedy(Variable(np.array([wid], dtype=np.int32)))
            next_wid = entity_vocab[EOS] if ( i == len(eline) - 1) else entity_vocab[eline[i+1]]
            tx = Variable(np.array([next_wid], dtype=np.int32))
            h = self.H(x_k)
            ct = mk_ct(gh, h.data)
            h2 = F.tanh(self.Wc1(ct) + self.Wc2(h))
            loss = F.softmax_cross_entropy(self.W(h2), tx)
            accum_loss += loss
            
        return accum_loss
    
    def reset_state(self):
        self.H.reset_state()


# In[ ]:

demb = 100
model = ATT(pv, ev, demb)
optimizer = optimizers.Adam()
optimizer.setup(model)

n_epoch = 20

print(len(src_lines))
import datetime

for epoch in range(n_epoch):
    sum_loss = 0
    
    for i in range(len(src_lines)-1):
        if i % 1000 == 0:
            print("{0}: Epoch {1} - Lines {2}...".format(datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S"), epoch, i))
        pln = src_lines[i].split()
        plnr = pln[::-1]
        eln = entity_lines[i].split()
        if len(eln) == 0:
            eln = [NIL]
#         print(eln)
        model.reset_state()
        model.zerograds()
        loss = model(plnr, eln)
        sum_loss += loss.data
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
    print("{0} finished".format(epoch), flush=True)
    print("train loss: {0}".format(sum_loss), flush=True)
    
    if epoch == n_epoch-1:
        outfile = "attention-"+str(n_epoch)+".model"
        serializers.save_npz(outfile, model)


# In[ ]:




# In[ ]:



