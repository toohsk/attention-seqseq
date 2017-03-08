
# coding: utf-8

# In[ ]:

from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Embedding, LSTM
from keras.preprocessing.text import text_to_word_sequence, one_hot
from keras.preprocessing import sequence

import random
from six.moves import range
import numpy as np
from gensim import corpora


# In[ ]:

def get_lines(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines    
        


# In[ ]:

amazon_cells_labelled_file = './data/sentiment labelled sentences/amazon_cells_labelled.txt'
yelp_labelled_file = './data/sentiment labelled sentences/yelp_labelled.txt'
imdb_labelled_file = './data/sentiment labelled sentences/imdb_labelled.txt'
lines = get_lines(imdb_labelled_file)


# In[ ]:

def cleaning(lines):
    
    positive_lines = []
    negative_lines = []
    max_token_size = -1
    
    for line in lines:
        elements = line.split('\t')
        # print(elements[0]) # Comment element.
        # print(elements[1]) # Sentiment element. 0 = Negative, 1 = Positive
        sentiment = int(elements[1]) 
        comment = elements[0] 
        tokens = [token.lower().strip(',.') for token in comment.split(' ')]

        # Updating most longest token size in dataset
        if len(tokens) > max_token_size:
            max_token_size = len(tokens)
        
        if sentiment == 1:
            positive_lines.append(tokens)
        elif sentiment == 0:
            negative_lines.append(tokens)
        else:
            print('Sentiment is not 0 or 1.')
            
    return positive_lines, negative_lines, max_token_size


# In[ ]:

positive_lines, negative_lines, input_length = cleaning(lines)
print(input_length)


# In[ ]:

dictionary = corpora.Dictionary(positive_lines)
dictionary.add_documents(negative_lines)

PADDING_WORD="<PADDING>"
dictionary.add_documents([[PADDING_WORD]])


# In[ ]:

n_token = len(dictionary.token2id)
padding_id = dictionary.token2id[PADDING_WORD]
batch_size = 32
nb_epoch = 30

print(n_token)
print(len(dictionary.token2id))


# In[ ]:

def convert2id(doc, dictionary):
    ids_doc = []
    for line in doc:
        ids = []
        for token in line:
            ids.append(dictionary[token])
        ids_doc.append(ids)
    
    return ids_doc


# In[ ]:

positive_lines_ids = convert2id(positive_lines, dictionary.token2id)


# In[ ]:

negative_lines_ids = convert2id(negative_lines, dictionary.token2id)


# In[ ]:

# pad each sentece by padding_word to set each setence to same dimension.
positive_lines_ids = sequence.pad_sequences(positive_lines_ids, maxlen=input_length, padding='post', value=padding_id)
negative_lines_ids = sequence.pad_sequences(negative_lines_ids, maxlen=input_length, padding='post', value=padding_id)


# In[ ]:

random.seed(42)

def random_sampling(positive_lines_ids, negative_lines_ids, reverse=False, sampling_rate=0.75):
    
    sampled_idxs = random.sample(range(len(positive_lines_ids)), int(len(positive_lines_ids)*sampling_rate))
    rest_idxs = list(set(range(len(positive_lines_ids))) - set(sampled_idxs))

    X_train = []
    y_train = []
    X_test = []
    y_test = []
                 
    for idx in sampled_idxs:
        if reverse:
            X_train.append(positive_lines_ids[idx][::-1])
            X_train.append(negative_lines_ids[idx][::-1])
        else:
            X_train.append(positive_lines_ids[idx])
            X_train.append(negative_lines_ids[idx])
        y_train.append(1)
        y_train.append(0)

    for idx in rest_idxs:
        if reverse:
            X_test.append(positive_lines_ids[idx][::-1])
            X_test.append(negative_lines_ids[idx][::-1])
        else:
            X_test.append(positive_lines_ids[idx])
            X_test.append(negative_lines_ids[idx])
        y_test.append(1)
        y_test.append(0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    return X_train, y_train, X_test, y_test


# In[ ]:


X_train, y_train, X_test, y_test = random_sampling(positive_lines_ids, negative_lines_ids)


# In[ ]:

print('Build model...')
model = Sequential()
model.add(Embedding(n_token, 100, dropout=0.2))
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:

print('Train...')

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


# In[ ]:

X_train, y_train, X_test, y_test = random_sampling(positive_lines_ids, negative_lines_ids, reverse=True)


# In[ ]:

print('Build model...')
model = Sequential()
model.add(Embedding(n_token, 100, dropout=0.2))
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:

print('Train...')

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

