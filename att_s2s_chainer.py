
# coding: utf-8

# In[ ]:

import numpy as np
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
import datetime
from filer3 import Filer
import glob
import random


# In[ ]:

from logging import getLogger, StreamHandler, INFO, FileHandler, Formatter

LOG_FILE='log/att_s2s_chainer.log'

logger = getLogger(__name__)
formatter = Formatter('[%(asctime)s] %(message)s')
stream_handler = StreamHandler()
stream_handler.setFormatter(formatter)

file_handler = FileHandler(LOG_FILE, 'a+')
file_handler.setFormatter(formatter)

logger.setLevel(INFO)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


# In[ ]:

logger.info("Set constants.")

EOS = '<eos>'
NIL = '<nil>'
UNKNOWN = '<unkn>'

EMBED_SIZE = 300
HIDDEN_SIZE = 150
BATCH_SIZE = 100
EPOCH_NUM = 50

FLAG_GPU = False


# In[ ]:

class LSTM_Encoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化
        :pram vocab_size: 語彙数
        :pram embed_size: 中間ベクトルのサイズ（デフォルト200）
        :pram hidden_size: 隠れ層のサイズ
        """
        super(LSTM_Encoder, self).__init__(
            xe = links.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh = links.Linear(embed_size, 4 * hidden_size),
            hh = links.Linear(hidden_size, 4 * hidden_size)
        )
        
    def __call__(self, x, c, h):
        """
        
        :pram x: one-hotな単語
        :pram c: 内部メモリ
        :pram h: 隠れ層
        :return: 次の内部メモリ， 次のかくれ層
        """
        e = functions.tanh(self.xe(x))
        return functions.lstm(c, self.eh(e) + self.hh(h))


# In[ ]:

class LSTM_Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化
        :param vocab_size: 語彙サイズ
        :param embed_size: 単語ベクトルのサイズ
        :param hidden_size: 中間ベクトルのサイズ
        """
        super(LSTM_Decoder, self).__init__(
            ye = links.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh = links.Linear(embed_size, 4 * hidden_size),
            hh = links.Linear(hidden_size, 4 * hidden_size),
            he = links.Linear(hidden_size, embed_size),
            ey = links.Linear(embed_size, vocab_size)
        )
        
    def __call__(self, y, c, h):
        """
        
        :pram y: one-hotな単語
        :pram c: 内部メモリ
        :pram h: 隠れ層
        :return: 予測単語，　次の内部メモリ，　次の隠れそう
        """        
        e = functions.tanh(self.ye(y))
        c, h = functions.lstm(c, self.eh(e) + self.hh(h))
        t = self.ey(functions.tanh(self.he(h)))
        return t, c, h


# In[ ]:

class Seq2Seq(Chain):
    def __init__(self, in_vocab_size, out_vocab_size, embed_size, hidden_size, batch_size, flag_gpu = False):
        """
        Seq2Seq の初期化
        :parm in_vocab_size: 入力層で扱う語彙数
        :parm out_vocab_size: 出力層で扱う語彙数
        :parm　embed_size: 特徴ベクトルのサイズ
        :parm hidden_size: 中間層のサイズ
        :parm batch_size: ミニバッチのサイズ
        :parm flag_gpu: GPUを使うかどうか
        """
        super(Seq2Seq, self).__init__(
            # Encoder のインスタンス化
            encoder = LSTM_Encoder(
                vocab_size=in_vocab_size,
                embed_size=embed_size,
                hidden_size=hidden_size
            ),
            # Decoder のインスタンス化
            decoder = LSTM_Decoder(
                vocab_size=out_vocab_size,
                embed_size=embed_size,
                hidden_size=hidden_size
            )
        )
        
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        if flag_gpu:
            self.ARR = cuda.cupy
        else:
            self.ARR = np
            
    def encode(self, words):
        """
        Encoderを計算する部分
        :param words: 単語のリスト
        :return:
        """
        # 内部メモリ，中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        
        # エンコーダに単語を順に読み込む
        for w in words:
            c, h = self.encoder(w, c, h)
            
        # 計算した中間ベクトルをデコーダーに引き継ぐためにインスタンス変数にする
        self.h = h
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        
    def decode(self, w):
        """
        デコーダーを計算する
        :param w: 単語
        :return: 単語数サイズのベクトルを出力
        """
        t, self.c, self.h = self.decoder(w, self.c, self.h)
        return t
    
    def reset(self):
        """
        中間ベクトル，　内部メモリ，　勾配の初期化
        :return:
        """
        self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        
        self.zerograds()


# In[ ]:

def forward(enc_words, dec_words, model, ARR):
    """
    順伝播の計算
    :param enc_words: 入力文の単語を記録したリスト
    :param dec_words: 出力文の単語を記録したリスト
    :param model: Seq2Seqのインスタンス
    :param ARR: cuda.cupyかnumpyか
    :return: 計算した損失の合計
    """
    batch_size = len(enc_words[0])
    model.reset()
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
    model.encode(enc_words)
    loss = Variable(ARR.zeros((), dtype='float32'))
    # <eos> をデコーダに読み込ませる
    t = Variable(ARR.array([0 for _ in range(batch_size)], dtype='int32'))
    
    # デコーダの計算
    for w in dec_words:
        y = model.decode(t)
        t = Variable(ARR.array(w, dtype='int32'))
        loss += functions.softmax_cross_entropy(y, t)
        
    return loss

def forward_test(enc_words, model, ARR):
    ret = []
    model.reset()
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
    model.encode(enc_words)
    t = Variable(ARR.array([0], dtype='int32'))
    counter = 0
    while counter < 50:
        y = model.decode(t)
        label = y.data.argmax()
        ret.append(label)
        t = Variable(ARR.array([label], dtype='int32'))
        counter += 1
        if label == 1:
            counter = 50
    return ret


# In[ ]:

def make_minibatch(src_lines, entity_lines, minibatch):
    # enc_words の作成
    enc_words = [src_lines[idx] for idx in minibatch]
    enc_max = np.max([len(row) for row in enc_words])
    enc_words = np.array([[-1]*(enc_max - len(row)) + row for row in enc_words], dtype='int32')
    enc_words = enc_words.T
    
    # dec_words の作成
    dec_words = [entity_lines[idx] for idx in minibatch]
    dec_max = np.max([len(row) for row in dec_words])
    dec_words = np.array([[-1]*(dec_max - len(row)) + row for row in dec_words], dtype='int32')
    dec_words = dec_words.T
    
    return enc_words, dec_words


# In[ ]:

def train():
    # 辞書の読み込み
#     word_to_id = Filer.readdump(DICT_PATH)
    src_lines, entity_lines, src_vocab, entity_vocab, id2wd, max_output_length = read_data_sources()
    
    # 語彙数
    in_vocab_size = len(src_vocab)
    out_vocab_size = len(entity_vocab)
    
    # モデルのインスタンス化
    model = Seq2Seq(
        in_vocab_size=in_vocab_size,
        out_vocab_size=out_vocab_size,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE, 
        batch_size=BATCH_SIZE,
        flag_gpu = FLAG_GPU
    )
    model.reset()
    
    # GPUのセット
    if FLAG_GPU:
        ARR = cuda.cupy
        cuda.get_device(0).use()
        model.to_gpu(0)
    else:
        ARR = np

    # 学習開始
    for epoch in range(EPOCH_NUM):
        # エポックごとにoptimizerの初期化
        opt = optimizers.Adam()
        opt.setup(model)
        opt.add_hook(optimizer.GradientClipping(5))
        
        data = list(range(len(src_lines)))
        random.shuffle(data)
        for num in range(len(data)//BATCH_SIZE):
            minibatch = data[num*BATCH_SIZE: (num+1)*BATCH_SIZE]
            # 読み込み用のデータ作成
            enc_words, dec_words = make_minibatch(src_lines, entity_lines, minibatch)
            # modelのリセット
            model.reset()
            # 順伝播
            total_loss = forward(enc_words=enc_words,
                                 dec_words=dec_words,
                                 model=model,
                                 ARR=ARR)
            # 学習
            logger.info("train loss: {0}".format(total_loss.data))
            total_loss.backward()
            opt.update()
            opt.zero_grads()

        logger.info('Epoch %s 終了' % (epoch+1))
        outputpath = './model/s2s_{0}_{1}_{2}_{3}'.format(EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE, epoch+1)
        serializers.save_hdf5(outputpath, model)


# In[ ]:

import pickle
def read_data_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
def create_word_dic(lines):
    word_dic = {}
    id2wd = {}
    for line in lines:
        lt = line.split()
        for w in lt:
            if w not in word_dic:
                vocab_size = len(word_dic)
                word_dic[w] = vocab_size
                id2wd[vocab_size] = w
                
    return word_dic, id2wd

def lines2words(lines, word_dict):
    token_lines = []
    for line in lines:
        token_id_list = []
        for token in line.split():
            token_id_list.append(word_dict[token])
        token_lines.append(token_id_list)
    return token_lines
    
def read_data_sources():
    logger.info("Loading data.")
    src_lines = read_data_from_pickle('./data/seq2seq_wiki/src/src_list_1.pickle').decode('utf8').split('\n')
    # dest_lines = read_data_from_pickle('./data/seq2seq_wiki/dest_all/replace_entity_list_1.pickle').decode('utf8').split('\n')
    entity_lines = read_data_from_pickle('./data/seq2seq_wiki/dest_entity/entity_list_with_nil_1.pickle').decode('utf8').split('\n')

    src_vocab, _ = create_word_dic(src_lines)
    src_vocab[EOS] = len(src_vocab)
    pv = len(src_vocab)
    
    max_output_length = 0
    for line in entity_lines:
        le = line.split()
        if len(le) > max_output_length:
            max_output_length = len(le)
    
    entity_vocab, id2wd = create_word_dic(entity_lines)

    if EOS not in entity_vocab:
        logger.info("Add <eos> to dictionary.")
        vocab_size = len(entity_vocab)
        entity_vocab[EOS] = vocab_size
        id2wd[vocab_size] = EOS

    if UNKNOWN not in entity_vocab:
        logger.info("Add <unkn> to dictionary.")
        vocab_size = len(entity_vocab)
        entity_vocab[UNKNOWN] = vocab_size
        id2wd[vocab_size] = UNKNOWN

    if NIL not in entity_vocab:
        logger.info("Add <nil> to dictionary.")
        vocab_size = len(entity_vocab)
        entity_vocab[NIL] = vocab_size
        id2wd[vocab_size] = NIL

    return lines2words(src_lines, src_vocab), lines2words(entity_lines, entity_vocab), src_vocab, entity_vocab, id2wd, max_output_length


# In[ ]:

username='Seq2Seq'

logger.info('Start training')
try:
    train()
except:
    logger.error('Error')
    raise

logger.info('Finish training')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

# def mk_ct(gh, ht):
#     alp = []
#     s = 0.0
#     for i in range(len(gh)):
#         s += np.exp(ht.dot(gh[i]))
#     ct = np.zeros(100)
#     for i in range(len(gh)):
#         alpi = np.exp(ht.dot(gh[i]))/s
#         ct += alpi * gh[i]
#     ct = Variable(np.array([ct]).astype(np.float32))
#     return ct


# In[ ]:

# class ATT(chainer.Chain):
#     def __init__(self, pv, ev, k):
#         super(ATT, self).__init__(
#             embedx = L.EmbedID(pv, k),
#             embedy = L.EmbedID(ev, k),
#             H = L.LSTM(k, k),
#             Wc1 = L.Linear(k, k),
#             Wc2 = L.Linear(k, k),
#             W = L.Linear(k, ev),
#         )
        
#     def __call__(self, pline, eline):
#         gh = []
#         for i in range(len(pline)):
#             wid = src_vocab[pline[i]]
#             x_k = self.embedx(Variable(np.array([wid], dtype=np.int32)))
#             h = self.H(x_k)
#             gh.append(np.copy(h.data[0]))
            
#         x_k = self.embedx(Variable(np.array([src_vocab[EOS]], dtype=np.int32)))
#         tx = Variable(np.array([entity_vocab[eline[0]]], dtype=np.int32))
#         h = self.H(x_k)
#         ct = mk_ct(gh, h.data[0])
#         h2 = F.tanh(self.Wc1(ct) + self.Wc2(h))
#         accum_loss = F.softmax_cross_entropy(self.W(h2), tx)
        
#         for i in range(len(eline)):
#             wid = entity_vocab[eline[i]]
#             x_k = self.embedy(Variable(np.array([wid], dtype=np.int32)))
#             next_wid = entity_vocab[EOS] if ( i == len(eline) - 1) else entity_vocab[eline[i+1]]
#             tx = Variable(np.array([next_wid], dtype=np.int32))
#             h = self.H(x_k)
#             ct = mk_ct(gh, h.data)
#             h2 = F.tanh(self.Wc1(ct) + self.Wc2(h))
#             loss = F.softmax_cross_entropy(self.W(h2), tx)
#             accum_loss += loss
            
#         return accum_loss
    
#     def reset_state(self):
#         self.H.reset_state()


# In[ ]:

# demb = 100
# model = ATT(pv, ev, demb)
# optimizer = optimizers.Adam()
# optimizer.setup(model)

# n_epoch = 50

# logger.info("line num: {}", len(src_lines))
# import datetime

# for epoch in range(n_epoch):
#     sum_loss = 0
    
#     for i in range(len(src_lines)-1):
#         if i % 1000 == 0:
#             logger.info("{0}: Epoch {1} - Lines {2}...".format(datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S"), epoch, i))
#             print("{0}: Epoch {1} - Lines {2}...".format(datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S"), epoch, i))
#         pln = src_lines[i].split()
#         plnr = pln[::-1]
#         eln = entity_lines[i].split()
#         if len(eln) == 0:
#             eln = [NIL]
# #         print(eln)
#         model.reset_state()
#         model.zerograds()
#         loss = model(plnr, eln)
#         sum_loss += loss.data
#         loss.backward()
#         loss.unchain_backward()
#         optimizer.update()
#     logger.info("{0} finished".format(epoch))
#     logger.info("train loss: {0}".format(sum_loss))
#     print("{0} finished".format(epoch), flush=True)
#     print("train loss: {0}".format(sum_loss), flush=True)
    
#     if epoch == n_epoch-1:
#         outfile = "attention-"+str(n_epoch)+".model"
#         serializers.save_npz(outfile, model)


# In[ ]:

# def mt(model, src_line, loop_limit=30):
#     lt = src_line.split()[::-1]
#     for i in range(len(lt)):
#         wid = src_vocab[lt[i]]
#         # print(lt[i])
#         x_k = model.embedx(Variable(np.array([wid], dtype=np.int32)))
#         h = model.H(x_k)
#     x_k = model.embedx(Variable(np.array([src_vocab[EOS]], dtype=np.int32)))
#     h = model.H(x_k)
#     wid = np.argmax(F.softmax(model.W(h)).data[0])
    
#     src_ary = []
#     if wid in id2wd:
#         src_ary.append(id2wd[wid])
#     else:
#         src_ary.append(UNKNOWN)
    
#     logger.info(' / '.join(src_ary))
# #     print(' / '.join(src_ary))
    
#     loop = 0
#     mt_ary = []
#     while (wid != entity_vocab[EOS]) and (loop <= loop_limit):
#         x_k = model.embedy(Variable(np.array([wid], dtype=np.int32)))
#         h = model.H(x_k)
#         wid = np.argmax(F.softmax(model.W(h)).data[0])
    
#         if wid in id2wd:
#             mt_ary.append(id2wd[wid])
#         else:
#             mt_ary.append(UNKNOWN)
            
#         loop += 1
# #     print(' / '.join(mt_ary))
#     logger.info(' / '.join(mt_ary))
#     logger.info(' ----------------- ')
#     print()


# In[ ]:

# print("max loop limit: {}".format(max_output_length))
# logger.info("max loop limit: {}".format(max_output_length))
# for line in src_lines:
# #     print(line)
#     logger.info(line)
#     mt(model, line, loop_limit=max_output_length+1)


# In[ ]:



