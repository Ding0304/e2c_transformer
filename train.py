import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from zh_converter import Converter
from nltk import word_tokenize
from torch.autograd import Variable

import nltk
import os

# 指定自定义下载路径
custom_download_dir = "./nltk/punkt"
# 确保目标文件夹存在
os.makedirs(custom_download_dir, exist_ok=True)
# 下载 'punkt' 到自定义路径 只下载一次
# nltk.download('punkt', download_dir=custom_download_dir)
# 添加自定义路径到 NLTK 的搜索路径
nltk.data.path.append("./nltk/punkt")

# 定义常量
PAD = 0                             # 填充占位符的索引
UNK = 1                             # 未登录词标识符的索引
BATCH_SIZE = 256                    # 批次大小
EPOCHS = 50                        # 训练轮数
LAYERS = 6                          # Transformer中Encoder、Decoder层数
H_NUM = 8                           # 多头注意力个数
D_MODEL = 256                       # 输入、输出词向量维数
D_FF = 1024                         # Feed Forward全连接层维数
DROPOUT = 0.1                       # Dropout比例
MAX_LENGTH = 60                     # 语句最大长度

TRAIN_FILE = 'datasets/train.txt'  # 训练集文件路径
DEV_FILE = "datasets/val.txt"      # 验证集文件路径
SAVE_FILE = 'output_weight/model1.pt'        # 模型保存路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择


def seq_padding(X, padding=PAD):
    """
    按批次（batch）对数据填充、长度对齐
    :param X: 输入的批次数据，一个二维列表
    :param padding: 填充值，默认为PAD
    :return: 填充后的二维数组
    """
    # 计算该批次各条样本语句长度
    L = [len(x) for x in X]
    # 获取该批次样本中语句长度最大值
    ML = max(L)
    # 遍历该批次样本，如果语句长度小于最大长度，则用padding填充
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def cht_to_chs(sent):
    """
    将繁体中文转换为简体中文
    :param sent: 输入的繁体中文句子
    :return: 转换后的简体中文句子
    """
    sent = Converter("zh-hans").convert(sent)
    sent.encode("utf-8")
    return sent


class PrepareData:
    def __init__(self, train_file, dev_file):
        """
        数据预处理类，负责加载数据、构建词表、划分批次等操作
        :param train_file: 训练集文件路径
        :param dev_file: 验证集文件路径
        """
        # 读取数据、分词
        self.train_en, self.train_cn = self.load_data(train_file)
        self.dev_en, self.dev_cn = self.load_data(dev_file)
        # 构建词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = \
            self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = \
            self.build_dict(self.train_cn)
        # 单词映射为索引
        self.train_en, self.train_cn = self.word2id(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.word2id(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)
        # 划分批次、填充、掩码
        self.train_data = self.split_batch(self.train_en, self.train_cn, BATCH_SIZE)
        self.dev_data = self.split_batch(self.dev_en, self.dev_cn, BATCH_SIZE)

    def load_data(self, path):
        """
        读取英文、中文数据，并对每条样本分词
        :param path: 数据文件路径
        :return: 英文和中文数据列表
        """
        en, cn = [], []
        with open(path, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                sent_en, sent_cn = line.strip().split("\t")
                sent_en = sent_en.lower()
                sent_cn = cht_to_chs(sent_cn)
                sent_en = ["BOS"] + word_tokenize(sent_en) + ["EOS"]
                # 中文按字符切分
                sent_cn = ["BOS"] + [char for char in sent_cn] + ["EOS"]
                en.append(sent_en)
                cn.append(sent_cn)
        return en, cn

    def build_dict(self, sentences, max_words=5e4):
        """
        构造分词后的列表数据，构建单词-索引映射
        :param sentences: 分词后的句子列表
        :param max_words: 最大保留单词数
        :return: 单词-索引字典、总单词数、索引-单词字典
        """
        # 统计数据集中单词词频
        word_count = Counter([word for sent in sentences for word in sent])
        # 按词频保留前max_words个单词构建词典
        ls = word_count.most_common(int(max_words))
        total_words = len(ls) + 2
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = UNK
        word_dict['PAD'] = PAD
        # 构建id2word映射
        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict

    def word2id(self, en, cn, en_dict, cn_dict, sort=True):
        """
        将英文、中文单词列表转为单词索引列表
        :param en: 英文单词列表
        :param cn: 中文单词列表
        :param en_dict: 英文单词-索引字典
        :param cn_dict: 中文单词-索引字典
        :param sort: 是否按英文语句长度排序
        :return: 英文、中文索引列表
        """
        out_en_ids = [[en_dict.get(word, UNK) for word in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(word, UNK) for word in sent] for sent in cn]

        # 按语句长度排序
        if sort:
            sorted_index = self.len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[idx] for idx in sorted_index]
            out_cn_ids = [out_cn_ids[idx] for idx in sorted_index]
        return out_en_ids, out_cn_ids

    def len_argsort(self, seq):
        """
        按语句长度排序，返回排序后原语句的索引下标
        :param seq: 语句列表
        :return: 排序后的索引下标
        """
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def split_batch(self, en, cn, batch_size, shuffle=True):
        """
        划分批次
        :param en: 英文索引列表
        :param cn: 中文索引列表
        :param batch_size: 批次大小
        :param shuffle: 是否随机打乱批次顺序
        :return: 批次列表
        """
        idx_list = np.arange(0, len(en), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)

        batches = []
        for idx in idx_list:
            # 起始索引最大的批次可能发生越界，需限定其索引
            batch_index = np.arange(idx, min(idx + batch_size, len(en)))
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            # 对当前批次中所有语句填充、对齐长度
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            # 将当前批次添加到批次列表
            batches.append(Batch(batch_en, batch_cn))
        return batches


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """
        初始化Embedding层
        :param d_model: 词向量维度
        :param vocab: 词汇表大小
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        前向传播，返回词向量
        :param x: 输入的单词索引
        :return: 词向量
        """
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        初始化位置编码层
        :param d_model: 词向量维度
        :param dropout: Dropout比例
        :param max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 位置编码矩阵，维度[max_len, embedding_dim]
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        position = torch.arange(0.0, max_len, device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=DEVICE) * (-math.log(10000.0) / d_model)).unsqueeze(0)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe.unsqueeze_(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播，将位置编码加到词向量上
        :param x: 输入的词向量
        :return: 加上位置编码后的词向量
        """
        x += Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

def clones(module, N):
    """
    克隆基本单元，克隆的单元之间参数不共享
    :param module: 要克隆的模块
    :param N: 克隆的数量
    :return: 包含N个克隆模块的ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """
    Scaled Dot-Product Attention（方程（4））
    :param query: Query张量，形状为(batch_size, h, seq_len, d_k)
    :param key: Key张量，形状为(batch_size, h, seq_len, d_k)
    :param value: Value张量，形状为(batch_size, h, seq_len, d_k)
    :param mask: 掩码张量，默认为None
    :param dropout: Dropout层，默认为None
    :return: 注意力加权后的Value张量和注意力权重矩阵
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention（编码器（2））
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    """
    层归一化
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(std ** 2 + self.eps)
        return self.a_2 * x + self.b_2


class SublayerConnection(nn.Module):
    """
    通过层归一化和残差连接，连接Multi-Head Attention和Feed Forward
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        x_ = self.norm(x)
        x_ = sublayer(x_)
        x_ = self.dropout(x_)
        return x + x_


class PositionwiseFeedForward(nn.Module):
    """
    Feed Forward层
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class EncoderLayer(nn.Module):
    """
    编码器基本单元
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """
    编码器
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    解码器基本单元
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """
    解码器
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    """
    解码器输出经线性变换和softmax函数映射为下一时刻预测单词的概率分布
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def subsequent_mask(size):
    """
    生成后续掩码矩阵，用于屏蔽未来信息
    :param size: 序列长度
    :return: 后续掩码矩阵
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# 示例：位置编码可视化保存到本地
emb_dim = 64
max_seq_len = 100
seq_len = 20

pe = PositionalEncoding(emb_dim, 0, max_seq_len)
positional_encoding = pe(torch.zeros(1, seq_len, emb_dim, device=DEVICE))

# 保存热力图
plt.figure()
sns.heatmap(positional_encoding.squeeze().to("cpu"))
plt.xlabel("i")
plt.ylabel("pos")
plt.savefig("imgs/positional_encoding_heatmap.png")
plt.close()

# 保存位置编码曲线图
plt.figure()
y = positional_encoding.to("cpu").numpy()
plt.plot(np.arange(seq_len), y[0, :, 0:64:8], ".")
plt.legend(["dim %d" % p for p in [0, 7, 15, 31, 63]])
plt.savefig("imgs/positional_encoding_plot.png")
plt.close()

# 保存后续掩码矩阵图
plt.figure(figsize=(5, 5))
plt.imshow(subsequent_mask(20)[0])
plt.savefig("imgs/subsequent_mask.png")
plt.close()

class Batch:
    """
    批次类：
        1. 输入序列（源）
        2. 输出序列（目标）
        3. 构造掩码
    """
    def __init__(self, src, trg=None, pad=PAD):
        # 将输入、输出单词id表示的数据规范成整数类型
        src = torch.from_numpy(src).to(DEVICE).long()
        trg = torch.from_numpy(trg).to(DEVICE).long() if trg is not None else None
        self.src = src
        # 对于当前输入的语句非空部分进行判断，bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对解码器使用的目标语句进行掩码
        if trg is not None:
            # 解码器使用的目标输入部分
            self.trg = trg[:, :-1]  # 去除最后一列
            # 解码器训练时应预测输出的目标结果
            self.trg_y = trg[:, 1:]  # 去除第一列的SOS
            # 将目标输入部分进行注意力掩码
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的目标结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        创建一个掩码以隐藏填充和未来单词。
        :param tgt: 目标张量
        :param pad: 填充符号
        :return: 掩码张量
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class Transformer(nn.Module):
    """
    Transformer模型
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        """
        编码输入序列
        :param src: 源序列
        :param src_mask: 源序列掩码
        :return: 编码结果
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        解码目标序列
        :param memory: 编码器输出
        :param src_mask: 源序列掩码
        :param tgt: 目标序列
        :param tgt_mask: 目标序列掩码
        :return: 解码结果
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Transformer整体前向传播
        :param src: 源序列
        :param tgt: 目标序列
        :param src_mask: 源序列掩码
        :param tgt_mask: 目标序列掩码
        :return: Transformer输出
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    构建Transformer模型
    :param src_vocab: 源词汇表大小
    :param tgt_vocab: 目标词汇表大小
    :param N: 编码器和解码器层数
    :param d_model: 输入、输出词向量维数
    :param d_ff: Feed Forward全连接层维数
    :param h: 多头注意力头数
    :param dropout: Dropout比例
    :return: Transformer模型
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model).to(DEVICE)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    position = PositionalEncoding(d_model, dropout).to(DEVICE)
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
        Generator(d_model, tgt_vocab)).to(DEVICE)

    # 初始化参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)


class LabelSmoothing(nn.Module):
    """
    标签平滑
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        计算标签平滑损失
        :param x: 预测分布
        :param target: 真实标签
        :return: 损失值
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    """
    简单的计算损失和进行参数反向传播更新训练的函数
    """
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        """
        计算损失并进行反向传播
        :param x: 模型输出
        :param y: 真实标签
        :param norm: 归一化因子
        :return: 损失值
        """
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()


class NoamOpt:
    """
    学习率调度器
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """
        更新参数和学习率
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        计算学习率
        :param step: 当前步数
        :return: 学习率
        """
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    """
    获取标准优化器
    :param model: Transformer模型
    :return: NoamOpt实例
    """
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# 示例：标签平滑可视化保存到本地
crit = LabelSmoothing(5, 0, 0.4)  # 设定一个ϵ=0.4
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0]])
v = crit(Variable(predict.log()), Variable(torch.LongTensor([2, 1, 0])))
plt.imshow(crit.true_dist)
plt.savefig("imgs/label_smoothing.png")
plt.close()

# 示例：学习率曲线保存到本地
opts = [NoamOpt(512, 1, 4000, None),
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])
plt.savefig("imgs/learning_rate_schedule.png")
plt.close()


def run_epoch(data, model, loss_compute, epoch):
    """
    运行一个epoch的训练或评估。
    :param data: 数据迭代器
    :param model: Transformer模型
    :param loss_compute: 损失计算函数
    :param epoch: 当前epoch编号
    :return: 平均损失值
    """
    start = time.time()
    total_tokens = 0.               # 总Token数
    total_loss = 0.                 # 总损失
    tokens = 0.                     # 当前批次Token计数

    for i, batch in enumerate(data):
        # 模型前向传播
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        # 计算损失
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        # 每50个批次打印一次进度
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch %d Batch: %d Loss: %f Tokens per Sec: %fs" % (
                epoch, i - 1, loss / batch.ntokens, (tokens / elapsed / 1000)))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


def plot_loss_curve(train_losses, dev_losses):
    """
    绘制训练和验证损失曲线，并保存图像。
    :param train_losses: 训练损失列表
    :param dev_losses: 验证损失列表
    """
    # 确保 imgs 文件夹存在
    if not os.path.exists("imgs"):
        os.makedirs("imgs")

    # 创建图像
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="blue")
    plt.plot(range(1, len(dev_losses) + 1), dev_losses, label="val Loss", color="orange")

    # 添加标题和标签
    plt.title("Loss Curve", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)

    # 保存图像
    plt.savefig("imgs/loss_curve1.png")
    plt.close()

    print("Loss curve saved to imgs/loss_curve.png")

def train(data, model, criterion, optimizer):
    """
    训练并保存模型。
    :param data: 数据对象，包含训练集和验证集
    :param model: Transformer模型
    :param criterion: 损失函数
    :param optimizer: 优化器
    """
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_dev_loss = 1e5

    # 添加用于记录训练和验证损失的列表
    train_losses = []
    dev_losses = []

    for epoch in range(EPOCHS):
        # 模型训练模式
        model.train()
        train_loss = run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)
        train_losses.append(train_loss)

        # 模型评估模式
        model.eval()
        print('>>>>> Evaluate')
        dev_loss = run_epoch(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        dev_losses.append(dev_loss)
        print('<<<<< Evaluate loss: %f' % dev_loss)

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss，则保存当前模型
        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), SAVE_FILE)
            best_dev_loss = dev_loss
            print('****** Save model done... ******')

        print()

    # 调用绘图函数
    plot_loss_curve(train_losses, dev_losses)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    贪心解码：传入一个训练好的模型，对指定数据进行预测。
    :param model: Transformer模型
    :param src: 源序列
    :param src_mask: 源序列掩码
    :param max_len: 最大生成长度
    :param start_symbol: 开始符号ID
    :return: 解码结果
    """
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)

    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))

        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])

        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

    return ys


def evaluate(data, model):
    """
    在data上用训练好的模型进行预测，打印模型翻译结果。
    :param data: 数据对象，包含英文和中文数据
    :param model: Transformer模型
    """
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for i in range(len(data.dev_en)):
            # 打印待翻译的英文语句
            en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
            print("\n" + en_sent)

            # 打印对应的中文语句答案
            cn_sent = " ".join([data.cn_index_dict[w] for w in data.dev_cn[i]])
            print("".join(cn_sent))

            # 将当前以单词id表示的英文语句数据转为tensor，并放如DEVICE中
            src = torch.from_numpy(np.array(data.dev_en[i])).long().to(DEVICE)
            src = src.unsqueeze(0)  # 增加一维
            src_mask = (src != 0).unsqueeze(-2)  # 设置attention mask

            # 用训练好的模型进行decode预测
            out = greedy_decode(model, src, src_mask, max_len=MAX_LENGTH, start_symbol=data.cn_word_dict["BOS"])

            # 初始化一个用于存放模型翻译结果语句单词的列表
            translation = []
            for j in range(1, out.size(1)):  # 注意：开始符"BOS"的索引0不遍历
                sym = data.cn_index_dict[out[0, j].item()]
                if sym != 'EOS':  # 如果输出字符不为'EOS'终止符，则添加到翻译结果列表
                    translation.append(sym)
                else:
                    break  # 否则终止遍历

            # 打印模型翻译输出的中文语句结果
            print("translation: %s" % " ".join(translation))


# 数据预处理
data = PrepareData(TRAIN_FILE, DEV_FILE)
src_vocab = len(data.en_word_dict)
tgt_vocab = len(data.cn_word_dict)
print("src_vocab %d" % src_vocab)
print("tgt_vocab %d" % tgt_vocab)

# 初始化模型
model = make_model(
    src_vocab,
    tgt_vocab,
    LAYERS,
    D_MODEL,
    D_FF,
    H_NUM,
    DROPOUT
)

# 训练
print(">>>>>>> start train")
train_start = time.time()
criterion = LabelSmoothing(tgt_vocab, padding_idx=0, smoothing=0.0)
optimizer = NoamOpt(D_MODEL, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

train(data, model, criterion, optimizer)
print(f"<<<<<<< finished train, cost {time.time() - train_start:.4f} seconds")

# 预测
# 加载模型
model.load_state_dict(torch.load(SAVE_FILE))
# 开始预测
print(">>>>>>> start evaluate")
evaluate_start = time.time()
evaluate(data, model)
print(f"<<<<<<< finished evaluate, cost {time.time() - evaluate_start:.4f} seconds")