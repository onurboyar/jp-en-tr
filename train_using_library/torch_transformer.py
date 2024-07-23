ANKI_LEXICON_PATH = 'Datasets/eng_jpn.txt'
KYOTO_LEXICON_PATH = 'Datasets/kyoto_lexicon.csv'

BATCH_SIZE = 20
EPOCHS = 100

D_MODEL = 512
HEADS = 8
N = 6

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import torchtext
from torchtext import data
from torchtext.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split
import math
import copy
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu
# from janome.tokenizer import Tokenizer

if torch.cuda.is_available():
    device = 'cuda:0'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
print(f'Running on device: {device}')


anki_dataset_df = pd.read_csv(ANKI_LEXICON_PATH,sep='\t',names=['Japanese','English']) 
kyoto_lexicon_df = pd.read_csv(KYOTO_LEXICON_PATH, on_bad_lines='warn')
anki_dataset_df.dropna(inplace=True)
kyoto_lexicon_df = kyoto_lexicon_df[['日本語', '英語']]
kyoto_lexicon_df.columns = ['Japanese', 'English']
kyoto_lexicon_df.dropna(inplace=True)
JA = spacy.blank('ja')
EN = spacy.load("en_core_web_sm")

def tokenize_ja(sentence):
    return [tok.text for tok in JA.tokenizer(sentence)]

def tokenize_en(sentence):
    return [tok.text for tok in EN.tokenizer(sentence)]


JA_TEXT = Field(tokenize=tokenize_ja) 
EN_TEXT = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>') 
frames = [kyoto_lexicon_df, anki_dataset_df]
merged_dataset_df = pd.concat(frames)
# train, val, test = train_val_test_split(kyoto_lexicon_df, test_size=0.3)
train, val, test = np.split(merged_dataset_df.sample(frac=1), [int(.6*len(merged_dataset_df)), int(.8*len(merged_dataset_df))])
train.to_csv('train.csv', index=False)
val.to_csv('val.csv', index=False) 
test.to_csv('test.csv', index=False) 

data_fields = [('Japanese', JA_TEXT), ('English', EN_TEXT)]

train, val, test = TabularDataset.splits(path='./',
                        train='train.csv', 
                        validation='val.csv',
                        test = 'test.csv',
                        format='csv',        
                        fields = data_fields )

JA_TEXT.build_vocab(train, val) 
EN_TEXT.build_vocab(train, val) 

train_iter = BucketIterator(
    train,
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.English),
    shuffle=True
)

batch = next(iter(train_iter)) 
print(batch.English)

global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.Japanese))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.English) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


train_iter = MyIterator(
    train,
    batch_size=1300,
    device=0,
    repeat=False,
    sort_key=lambda x: (len(x.Japanese), len(x.English)),
    batch_size_fn=batch_size_fn,
    train=True,
    shuffle=True
)


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        # print("inside embedder init")
        super(Embedder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        # print("inside embedder forward")
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        # print("inside PositionalEncoder init")
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # print("inside PositionalEncoder forward")
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], \
        requires_grad=False).to(device)
        # print(x)
        return x


# def create_masks(src, trg):
#     src_mask = (src != JA_TEXT.vocab.stoi['<pad>']).unsqueeze(1).unsqueeze(2)
#     trg_pad_mask = (trg != EN_TEXT.vocab.stoi['<pad>']).unsqueeze(1).unsqueeze(2)
#     size = trg.size(1)
#     nopeak_mask = torch.triu(torch.ones((size, size), device=device) == 1).transpose(0, 1)
#     nopeak_mask = nopeak_mask.float().masked_fill(nopeak_mask == 0, float('-inf')).masked_fill(nopeak_mask == 1, float(0.0))
#     trg_mask = trg_pad_mask & nopeak_mask.bool()
#     print(src_mask.size())
#     print(trg_mask.size())

#     return src_mask, trg_mask

def create_masks(src, trg):
    src_pad = JA_TEXT.vocab.stoi['<pad>']
    trg_pad = EN_TEXT.vocab.stoi['<pad>']

    # Source mask
    src_mask = (src != src_pad).unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1, src_seq_len]
    src_mask = src_mask.to(device)

    # Target padding mask
    trg_pad_mask = (trg != trg_pad) #.unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1, trg_seq_len]

    # No peak mask
    size = trg.size(1)  # Get the target sequence length
    nopeak_mask = torch.triu(torch.ones((size, size), device=device) == 1).transpose(0, 1)
    nopeak_mask = nopeak_mask.float().masked_fill(nopeak_mask == 0, float('-inf')).masked_fill(nopeak_mask == 1, float(0.0))

    # Combine padding mask and no peak mask
    print(trg_pad_mask.size())
    print(nopeak_mask.size())
    trg_mask = trg_pad_mask & nopeak_mask.bool()  # Shape: [batch_size, trg_seq_len, trg_seq_len]

    return src_mask, trg_mask



class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        # print("inside MultiHeadAttention __init__")
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        # print("inside MultiHeadAttention forward")
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        # print(output)
        return output

def attention(q, k, v, d_k, mask=None, dropout=None):
    # print("inside attention")
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    # print(scores)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    # print(output)
    return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        # print("inside FeedForward init")
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) 
    def forward(self, x):
        # print("inside FeedForward forward")
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        # print(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        # print("inside Norm init")
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        # print("inside Norm forward")
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        # print(norm)
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        # print("inside EncoderLayer init")
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # print("inside EncoderLayer forward")
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        # print("inside EncoderLayer forward : "+x)
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        # print("inside EncoderLayer __init__")
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).to(device)

    def forward(self, x, e_outputs, src_mask, trg_mask):
            # print("inside EncoderLayer forward")
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
            src_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff(x2))
            # print(x)
            return x

# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        # print("inside Encoder __init__")
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        # print("inside Encoder forward")
        x = self.embed(src)
        x = self.pe(x)
        for i in range(N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        # print("inside Decoder __init__")
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        # print("inside Decoder forward")
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.positional_encoding = PositionalEncoder(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        
    def forward(self, src, trg, src_mask, trg_mask):
        src = self.src_embedding(src) * math.sqrt(D_MODEL)
        trg = self.trg_embedding(trg) * math.sqrt(D_MODEL)
        src = self.positional_encoding(src)
        trg = self.positional_encoding(trg)
        src = src.transpose(0, 1)
        trg = trg.transpose(0, 1)
        transformer_out = self.transformer(src, trg, src_key_padding_mask=src_mask, tgt_mask=trg_mask)
        output = self.fc_out(transformer_out)
        return output.transpose(0, 1)


src_vocab_size = len(JA_TEXT.vocab)
trg_vocab_size = len(EN_TEXT.vocab)

model = TransformerModel(src_vocab_size, trg_vocab_size, D_MODEL, HEADS, N, N, 2048).to(device)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

def train_model(model, epochs, print_every=50):
    model.train()
    start = datetime.now()
    temp = start
    total_loss = 0
    
    for epoch in range(epochs):
        for i, batch in enumerate(train_iter):
            src = batch.Japanese.transpose(0, 1).to(device)
            trg = batch.English.transpose(0, 1).to(device)
            trg_input = trg[:, :-1]
            targets = trg[:, 1:].contiguous().view(-1)
            src_mask, trg_mask = create_masks(src, trg_input)
            
            optim.zero_grad()
            preds = model(src, trg_input, src_mask, trg_mask)
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets, ignore_index=EN_TEXT.vocab.stoi['<pad>'])
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                perplexity  = math.exp(loss_avg)
                print(f"time = {datetime.now() - start}, epoch {epoch + 1}, iter = {i + 1}, loss = {loss_avg}, perplexity = {perplexity}, {datetime.now() - temp} per {print_every} iters")
                total_loss = 0
                temp = datetime.now()
    
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
        }, f'model_2.pth')

model.to(device)
train_model(model, EPOCHS)

