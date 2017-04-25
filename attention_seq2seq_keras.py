
# coding: utf-8

# In[ ]:

import numpy as np
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from six.moves import range


# In[ ]:

class CharTable(object):
    
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen
        
    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X
    
    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


# In[ ]:

class Colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


# In[ ]:

TRAIN_SIZE = 50000
DIGITS = 3
INVERT = True

RNN = recurrent.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
MAXLEN = DIGITS + 2 + DIGITS

chars = '0123456789+='
ctable = CharTable(chars, MAXLEN)

questions = []
expected = []
seen = set()


# In[ ]:

get_ipython().run_cell_magic('time', '', "\n# Generate data\nwhile len(questions) < TRAIN_SIZE:\n    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS+1))))\n    a, b = f(), f()\n    \n    key = tuple(sorted((a,b)))\n    if key in seen:\n        continue\n        \n    seen.add(key)\n    q = '{}+{}='.format(a, b)\n    query = q + ''*(MAXLEN - len(q))\n    \n    ans = str(a+b)\n    ans += ''*(DIGITS + 1 - len(ans))\n    \n    if INVERT:\n        query = query[::-1]\n        \n    questions.append(query)\n    expected.append(ans)\n    \nprint(questions[:10])\nprint(expected[:10])\nprint('Total addition questions:', len(questions))")


# In[ ]:

print('Vectorization...')
X = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    X[i] = ctable.encode(sentence, maxlen=MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, maxlen=DIGITS + 1)

# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
indices = np.arange(len(y))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over
split_at = int(len(X) - len(X) / 10)
(X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
(y_train, y_val) = (y[:split_at], y[split_at:])


# In[ ]:

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


# In[ ]:

get_ipython().run_cell_magic('time', '', '\n#Training the model with the encoded inputs\nprint(\'Build model...\')\nmodel = Sequential()\n# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE\n# note: in a situation where your input sequences have a variable length,\n# use input_shape=(None, nb_feature).\nmodel.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))\n# For the decoder\'s input, we repeat the encoded input for each time step\nmodel.add(RepeatVector(DIGITS + 1))\n# The decoder RNN could be multiple layers stacked or a single layer\nfor _ in range(LAYERS):\n    model.add(RNN(HIDDEN_SIZE, return_sequences=True))\n\n# For each of step of the output sequence, decide which character should be chosen\nmodel.add(TimeDistributed(Dense(len(chars))))\nmodel.add(Activation(\'softmax\'))')


# In[ ]:

get_ipython().run_cell_magic('time', '', "\nmodel.compile(loss='categorical_crossentropy',\n              optimizer='adam',\n              metrics=['accuracy'])\n\n# Train the model each generation and show predictions against the validation dataset\nfor iteration in range(1, 20):\n    print()\n    print('-' * 50)\n    print('Iteration', iteration)\n    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,\n              validation_data=(X_val, y_val))\n    \n    score = model.evaluate(X_val, y_val, verbose=0)\n    print('\\n')\n    print('Test score:', score[0])\n    print('Test accuracy:', score[1])\n    print('\\n')")


# In[ ]:

get_ipython().run_cell_magic('time', '', "\n#For predicting the outputs, the predict method will return \n#an one hot encoded ouput, we decode the one hot encoded \n#ouptut to get our final output\n\n# Select 10 samples from the validation set at random so we can visualize errors\nfor i in range(20):\n    ind = np.random.randint(0, len(X_val))\n    rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]\n    preds = model.predict_classes(rowX, verbose=0)\n    q = ctable.decode(rowX[0])\n\n    correct = ctable.decode(rowy[0])\n    guess = ctable.decode(preds[0], calc_argmax=False)\n    print('Q', q[::-1] if INVERT else q)\n    print('T', correct)\n    print(Colors.ok + '☑' + Colors.close if correct == guess else Colors.fail + '☒' + Colors.close, guess)\n    print('---')")


# In[ ]:

for i in range(10):
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS+1))))
    a, b = f(), f()
    
    key = tuple(sorted((a,b)))
    if key in seen:
        continue
        
    seen.add(key)
    q = '{}+{}='.format(a, b)
    query = q + ''*(MAXLEN - len(q))
    
    ans = str(a+b)
    ans += ''*(DIGITS + 1 - len(ans))
    
    if INVERT:
        query = query[::-1]
        
    Xt = np.zeros((len(query), MAXLEN, len(chars)), dtype=np.bool)
    yt = np.zeros((len(query), DIGITS + 1, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(query):
        Xt[i] = ctable.encode(sentence, maxlen=MAXLEN)
    for i, sentence in enumerate(ans):
        yt[i] = ctable.encode(sentence, maxlen=DIGITS + 1)
    
    Xt = np.array(Xt)
    yt = np.array(yt)
    preds = model.predict_classes(Xt, verbose=0)
    q = ctable.decode(yt[0])

    correct = ctable.decode(yt[0])
    guess = ctable.decode(preds[0], calc_argmax=False)
    print('Q', q[::-1] if INVERT else q)
    print('T', correct)
    print(Colors.ok + '☑' + Colors.close if correct == guess else Colors.fail + '☒' + Colors.close, guess)
    print('---')


# In[ ]:




# In[ ]:



