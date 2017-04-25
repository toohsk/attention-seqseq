
# coding: utf-8

# In[ ]:

import re
import urllib
import MeCab
import pickle
from datetime import datetime


# In[ ]:

with open('./data/jawiki-latest-docs.xml', 'r') as f:
    lines = f.readlines()


# In[ ]:

len(lines)


# In[ ]:

lines[1]


# In[ ]:

DOCS_TAB = ('<docs>', '</docs>')

page_id_tab_pattern = re.compile(r'<doc\sid="(?P<id>[0-9]+)"\surl="(?P<url>.+?)"\stitle="(?P<title>.+?)">')
page_end_tab_pattern = re.compile(r'</doc>')
a_tab_pattern = re.compile(r'<a\shref="(?P<entity>.+?)">(?P<anchor_text>.+?)</a>')
ref_tab_pattern = re.compile(r'<ref.+?</ref>')

tagger = MeCab.Tagger('-d /usr/local/Cellar/mecab/0.996/lib/mecab/dic/mecab-ipadic-neologd')


# In[ ]:

def get_tokens(line):
    token_list = []
    node = tagger.parseToNode(line)
    while node:
        token_list.append(node.surface)
        node = node.next
    return token_list


# In[ ]:

def remove_empty_char(words):
    while words.count("") > 0:
        words.remove("")
    return words


# In[ ]:

len(lines)


# In[ ]:

def save_list_data(list_data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(list_data.encode('utf-8'), f)
        
# save_list_data(all_src, 'src_list.pickle')
# save_list_data(all_dest, 'replace_entity_list.pickle')
# save_list_data(all_only_entity, 'entity_list.pickle')


# In[ ]:

page_entity_dic = {}
skip_flag = False

all_src, all_dest, all_only_entity = [], [], []
src, dest, only_entity = [], [], []

NIL = '<nil>'

for i, line in enumerate(lines):
    if i % 10000 == 0:
        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), 'Executing: {0}...'.format(i))
    if i != 0 and i % 10000 == 0:
        idx = int(i / 10000)
#         print(all_src)
        save_list_data('\n'.join(all_src), './data/seq2seq_wiki/src/src_list_{0}.pickle'.format(idx))
        save_list_data('\n'.join(all_dest), './data/seq2seq_wiki/dest_all/replace_entity_list_{0}.pickle'.format(idx))
        save_list_data('\n'.join(all_only_entity), './data/seq2seq_wiki/dest_entity/entity_list_with_nil_{0}.pickle'.format(idx))
        all_src, all_dest, all_only_entity = [], [], []
    
    line = line.replace('\n', '')
    if line in DOCS_TAB:
        continue
    
    if page_end_tab_pattern.match(line):
#         print(page_title, "end")
#         print("Dictionary:", page_entity_dic)
        page_entity_dic = {}
        src, dest, only_entity = [], [], []
        continue
        
#     if ref_tab_pattern.match(line):
#         print(line)
#         print(ref_tab_pattern.sub(line, ""))
#         break
    
    page_id_match_obj = page_id_tab_pattern.search(line)
    if page_id_match_obj:
        page_title = page_id_match_obj.group('title')
        page_entity_dic[page_title] = page_title
    
    linked_line_match_obj = a_tab_pattern.findall(line)
    if linked_line_match_obj:
        for match_obj in linked_line_match_obj:
            entity = match_obj[0]
            anchor_text = match_obj[1]
            if entity.startswith("http"): continue
            page_entity_dic[anchor_text] = entity
    
    splited_line_match_obj = a_tab_pattern.split(line)
    if len(splited_line_match_obj) > 2:
        src, dest, only_entity = [], [], []
        for idx in range(0, len(splited_line_match_obj)):
            if splited_line_match_obj[idx] == "" or skip_flag:
                skip_flag = False
                continue
                
#             print(splited_line_match_obj[idx])
#             if splited_line_match_obj[idx].startswith("%") and (idx+1) <  len(splited_line_match_obj):
            current_token = splited_line_match_obj[idx]
            if current_token.startswith("%") and urllib.parse.unquote(current_token) != current_token:
                try:
                    encoded_token = splited_line_match_obj[idx+1]
                    src.append(encoded_token)
                    dest.append('['+urllib.parse.unquote(encoded_token)+']')
                    only_entity.append(urllib.parse.unquote(encoded_token))
                    skip_flag = True
                except Exception as e:
                    print(current_token)
                    print(splited_line_match_obj[idx+1])
            else:
                nouns = get_tokens( splited_line_match_obj[idx])
                src.extend(nouns)
                dest.extend(nouns)
                only_entity.extend([NIL for i in nouns])
        
        src = remove_empty_char(src)
        dest = remove_empty_char(dest)
        only_entity = remove_empty_char(only_entity)

#         print(src) 
#         print(dest)
#         print(only_entity)
#         print("Size src: {}, dest: {}".format(len(src), len(dest)))
        
        all_src.append(' '.join(src))
        all_dest.append(' '.join(dest))
        all_only_entity.append(' '.join(only_entity))
#     if i>100:
#         break


# In[ ]:

# print(all_src) 
# print(all_dest)
# print(all_only_entity)


# In[ ]:




# In[ ]:



