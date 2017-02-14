
# coding: utf-8

# In[ ]:

import MeCab
m = MeCab.Tagger ("-d /usr/local/Cellar/mecab/0.996/lib/mecab/dic/mecab-ipadic-neologd")


# In[ ]:

with open('./test_input.txt', 'r') as file:
    lines = file.readlines()


# In[ ]:

all_lines = []
for line in lines:
    split_line = []
    node = m.parseToNode(line.replace(" ", ""))
    while node:
        split_line.append(node.surface)
        node = node.next
    print(' '.join(split_line[1:len(split_line)-1]))


# In[ ]:

all_lines


# In[ ]:



