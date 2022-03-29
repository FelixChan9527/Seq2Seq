import spacy
from torchtext.legacy.datasets import Multi30k     # 加载自带的数据集
from torchtext.legacy.data import Field, BucketIterator
import torch.nn as nn
import torch
import random
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

spacy_eng = spacy.blank('en')
spacy_ger = spacy.blank('de')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def tokenize_eng(text):     # 创建分词器
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def tokenize_ger(text):     # 创建分词器
    return [tok.text for tok in spacy_ger.tokenizer(text)]

english = Field(tokenize=tokenize_eng, lower=True, 
                init_token='<sos>', eos_token='<eos>')  # 设定起始标志位，终止标志位
german = Field(tokenize=tokenize_ger, lower=True, 
                init_token='<sos>', eos_token='<eos>')

train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(german, english))

english.build_vocab(train_data, max_size=10000, min_freq=2)     # 只取出现频率大于2的词，其余忽略
german.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        # 这里的hidden_size就是外部状态的大小， num_layers表示多少个LSTM堆叠
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):       # x的形状为(seq_lenth, batchsize)
        # embedding层的输出形状为(seq_lenth, batchsize, embedding_size)
        embedding = self.dropout(self.embedding(x))     
        # hidden为外部状态，cell为内部状态
        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, 
                        output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # 当前的x是编码器的输出，即当前时间步的序列元素xt，所有大小为[batch_size]
        # 但是需要进行维度扩展为[1, batchsize]，因为LSTM的输入必须为[length, batchsize]
        x = x.unsqueeze(0)

        # [1, batchsize, embedding_size]
        embedding = self.dropout(self.embedding(x))

        # output：[1, batchsize, hiddensize]
        out_puts, (hidden, cell) = self.rnn(embedding, (hidden, cell))

        # 将此时间步输出的xt映射为实际翻译输出的某个词
        # 大小为[1, N, length_of_vocab]，length_of_vocab即输出为单词表的编码长度
        predictions = self.fc(out_puts)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(DEVICE)

        hidden, cell = self.encoder(source)

        x = target[0]   # 获取开始标志位
        
        for t in range(1, target_len):
            # 输入上一时刻的输出，外部状态、内部状态
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output     # 当前句子所对应翻译的单词
            best_guess = output.argmax(1)   # 获取预测单词表中最有可能的单词索引
            
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

################## 训练 ##################

num_epochs = 20
learning_rate = 0.001
batch_size = 64

load_model = False
device = torch.device(DEVICE)
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

writer = SummaryWriter("/home/MyServer/My_Code/MachineLearning/en2de/logfiles")
step = 0

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
                                    (train_data, validation_data, test_data), 
                                    batch_size=batch_size, 
                                    sort_within_batch=True,
                                    sort_key=lambda x: len(x.src), 
                                    device=device)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, 
                        hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, 
                        hidden_size, output_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = english.vocab.stoi['<pad>']   # 忽略该字符
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

model_path = "/home/MyServer/My_Code/MachineLearning/en2de/en2de.pth"

for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_iterator), total=len(train_iterator), leave=False)
    for batch_idx, batch in loop:
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target)
        # 因为第0个是起始标志位，所以要从1开始
        # 输出的大小为(trg_len, batch_size, output_dim)，
        # 为了输入交叉熵函数的需要，则把trg_len和batch_size合并为一个维度，则
        # 输出变为一个二维数据，大小为[trg_len*batch_size, output_dim]
        output = output[1:].reshape(-1, output.shape[2])    
        target = target[1:].reshape(-1)     # 同样变为与输出相同的形式
        
        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()
        # clip_grad_norm_ 梯度截断
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())
    
    torch.save(model.state_dict(), model_path)










