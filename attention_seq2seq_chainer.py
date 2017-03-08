
# coding: utf-8

# In[ ]:

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


# In[ ]:

plain_vocab = {}
EOS = '<eos>'
entity_vocab = {}
id2wb = {}


# In[ ]:

plain_lines = open('./input.txt').read().split('\n')


# In[ ]:

print(plain_lines)
for line in plain_lines:
    lt = line.split()
    for w in lt:
        if w not in plain_vocab:
            plain_vocab[w] = len(plain_vocab)


# In[ ]:

plain_vocab[EOS] = len(plain_vocab)-1
pv = len(plain_vocab)


# In[ ]:

entity_lines = open('./output.txt').read().split('\n')


# In[ ]:

for line in entity_lines:
    lt = line.split()
    for w in lt:
        if w not in entity_vocab:
            id = len(entity_vocab)
            entity_vocab[w] = id
            id2wb[id] = w


# In[ ]:

id = len(entity_vocab)
entity_vocab[EOS] = id-1
id2wb[id-1] = EOS
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
            wid = plain_vocab[pline[i]]
            x_k = self.embedx(Variable(np.array([wid], dtype=np.int32)))
            h = self.H(x_k)
            gh.append(np.copy(h.data[0]))
            
        x_k = self.embedx(Variable(np.array([plain_vocab[EOS]], dtype=np.int32)))
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

sum_loss = 0

for epoch in range(100):
    for i in range(len(plain_lines)-1):
        
        pln = plain_lines[i].split()
        plnr = pln[::-1]
        eln = entity_lines[i].split()
        model.reset_state()
        model.zerograds()
        loss = model(plnr, eln)
        sum_loss += loss.data
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
        print(i, " finished")
        print("train loss:", sum_loss)
    
    outfile = "attention-"+str(epoch)+".model"
    serializers.save_npz(outfile, model)


# In[ ]:



