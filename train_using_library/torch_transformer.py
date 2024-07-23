import torch
from torch import nn, optim
import pandas as pd
import spacy
from torchtext.data import Field, BucketIterator, TabularDataset
import numpy as np
import math
from datetime import datetime
import torch.nn.functional as F

# Paths
ANKI_LEXICON_PATH = 'Datasets/eng_jpn.txt'
KYOTO_LEXICON_PATH = 'Datasets/kyoto_lexicon.csv'

# Hyperparameters
BATCH_SIZE = 20
EPOCHS = 100
D_MODEL = 512
HEADS = 8
N = 6

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Load datasets
anki_dataset_df = pd.read_csv(ANKI_LEXICON_PATH, sep='\t', names=['Japanese', 'English'])
kyoto_lexicon_df = pd.read_csv(KYOTO_LEXICON_PATH, on_bad_lines='warn')
anki_dataset_df.dropna(inplace=True)
kyoto_lexicon_df = kyoto_lexicon_df[['日本語', '英語']]
kyoto_lexicon_df.columns = ['Japanese', 'English']
kyoto_lexicon_df.dropna(inplace=True)

# Tokenizers
JA = spacy.blank('ja')
EN = spacy.load("en_core_web_sm")

def tokenize_ja(sentence):
    return [tok.text for tok in JA.tokenizer(sentence)]

def tokenize_en(sentence):
    return [tok.text for tok in EN.tokenizer(sentence)]

# Fields
JA_TEXT = Field(tokenize=tokenize_ja, lower=True, init_token='<sos>', eos_token='<eos>', pad_token='<pad>')
EN_TEXT = Field(tokenize=tokenize_en, lower=True, init_token='<sos>', eos_token='<eos>', pad_token='<pad>')

# Merging datasets
frames = [kyoto_lexicon_df, anki_dataset_df]
merged_dataset_df = pd.concat(frames)
train, val, test = np.split(merged_dataset_df.sample(frac=1), [int(.6*len(merged_dataset_df)), int(.8*len(merged_dataset_df))])

# Saving datasets to CSV
train.to_csv('train.csv', index=False)
val.to_csv('val.csv', index=False) 
test.to_csv('test.csv', index=False) 

# Data fields
data_fields = [('Japanese', JA_TEXT), ('English', EN_TEXT)]

# Creating datasets
train, val, test = TabularDataset.splits(path='./', train='train.csv', validation='val.csv', test='test.csv', format='csv', fields=data_fields)

# Building vocabularies
JA_TEXT.build_vocab(train, val)
EN_TEXT.build_vocab(train, val)

# Creating iterators
train_iter = BucketIterator(train, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.English), shuffle=True, device=device)
val_iter = BucketIterator(val, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.English), shuffle=False, device=device)
test_iter = BucketIterator(test, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.English), shuffle=False, device=device)

# Helper functions
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

def pad_sequence(seq, max_len, pad_value):
    if seq.size(1) < max_len:
        pad_size = max_len - seq.size(1)
        seq = F.pad(seq, (0, pad_size), value=pad_value)
    return seq

def create_masks(src, trg, src_pad, trg_pad, max_len):
    src = pad_sequence(src, max_len, src_pad)
    trg = pad_sequence(trg, max_len, trg_pad)

    print(f"src size: {src.size()}")  # Debugging statement
    print(f"trg size: {trg.size()}")  # Debugging statement

    src_mask = (src != src_pad).unsqueeze(1)  # Shape: [batch_size, 1, src_seq_len]
    trg_pad_mask = (trg != trg_pad).unsqueeze(1)  # Shape: [batch_size, 1, trg_seq_len]
    size = trg.size(1)
    nopeak_mask = torch.triu(torch.ones((size, size), device=device) == 1).transpose(0, 1)
    nopeak_mask = nopeak_mask.float().masked_fill(nopeak_mask == 0, float('-inf')).masked_fill(nopeak_mask == 1, float(0.0))
    trg_mask = trg_pad_mask & nopeak_mask.bool()  # Shape: [batch_size, trg_seq_len, trg_seq_len]

    src_mask = src_mask.squeeze(1)  # Shape: [batch_size, 1, src_seq_len]

    return src_mask, trg_mask, src, trg

def find_max_len(data_iter):
    max_src_len = 0
    max_trg_len = 0
    for batch in data_iter:
        src = batch.Japanese.transpose(0, 1)
        trg = batch.English.transpose(0, 1)
        max_src_len = max(max_src_len, src.size(1))
        max_trg_len = max(max_trg_len, trg.size(1))
    return max_src_len, max_trg_len

max_src_len, max_trg_len = find_max_len(train_iter)
MAX_SEQ_LEN = max(max_src_len, max_trg_len)
print(f'Max Sequence Length: {MAX_SEQ_LEN}')

class TransformerModel(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_len):
        super(TransformerModel, self).__init__()
        self.src_embed = nn.Sequential(Embedder(src_vocab, d_model), PositionalEncoder(d_model, max_seq_len))
        self.trg_embed = nn.Sequential(Embedder(trg_vocab, d_model), PositionalEncoder(d_model, max_seq_len))
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
        self.out = nn.Linear(d_model, trg_vocab)
        
    def forward(self, src, trg, src_mask, trg_mask):
        src = self.src_embed(src).transpose(0, 1)
        trg = self.trg_embed(trg).transpose(0, 1)
        output = self.transformer(src, trg, src_key_padding_mask=src_mask, tgt_mask=trg_mask)
        output = self.out(output)
        return output

# Model parameters
src_vocab_size = len(JA_TEXT.vocab)
trg_vocab_size = len(EN_TEXT.vocab)
model = TransformerModel(src_vocab_size, trg_vocab_size, D_MODEL, HEADS, N, N, 2048, MAX_SEQ_LEN).to(device)

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
def train_model(model, epochs):
    model.train()
    src_pad = JA_TEXT.vocab.stoi['<pad>']
    trg_pad = EN_TEXT.vocab.stoi['<pad>']
    
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(train_iter):
            src = batch.Japanese.transpose(0, 1).to(device)
            trg = batch.English.transpose(0, 1).to(device)
            trg_input = trg[:, :-1]
            targets = trg[:, 1:].contiguous().view(-1)
            src_mask, trg_mask, src, trg_input = create_masks(src, trg_input, src_pad, trg_pad, MAX_SEQ_LEN)
            
            optimizer.zero_grad()
            preds = model(src, trg_input, src_mask, trg_mask)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets, ignore_index=trg_pad)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_iter)}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss/len(train_iter):.4f}')
        torch.save(model.state_dict(), f'transformer_epoch_{epoch+1}.pth')

model.to(device)
train_model(model, EPOCHS)
