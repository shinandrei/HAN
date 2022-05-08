


import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
def sequence_mask(lengths, max_len=None):
    lengths_shape = lengths.shape  # torch.size() is a tuple
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1))).reshape(lengths_shape)

class HAN(nn.Module):
    def __init__(self, wordvec, hidden_dim, num_layers, dropout=0.5):
        super(HAN, self).__init__()
        #Embeding Layer
        vocab_size = wordvec.shape[0]
        self.embedding_dim = wordvec.shape[1]

        self.embedding = torch.nn.Embedding(vocab_size, self.embedding_dim, _weight=Parameter(torch.tensor(wordvec)))
     #   self.embedding.weight = torch.nn.Parameter(torch.tensor(wordvec))
        self.dropout = nn.Dropout(dropout)

        #News level attention
        self.t = nn.Sequential(nn.Linear(self.embedding_dim, 1), nn.Sigmoid())
        #Words level attention
        self.u = nn.Sequential(nn.Linear(self.embedding_dim, 1), nn.Sigmoid())

        #Sequence modeling

        self.gru = nn.GRU(self.embedding_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True)
        # nn.init.xavier_normal_(self.gru.weight_ih_l0)
        # nn.init.xavier_normal_(self.gru.weight_hh_l0)
        #Temporal attention
        self.o = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())

        #Discriminative Network (MLP)
        self.fc0 = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ELU())
        self.fc1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU())

        # StockNet: 2-class
        self.fc_out = nn.Linear(hidden_dim, 3)

    def forward(self, x, day_len, news_len, training=True):

        max_dlen = int(torch.max(day_len).numpy())
        max_nlen = int(torch.max(news_len).numpy())
        x = x[:, :, :max_dlen, :max_nlen]
        news_len = news_len[:, :, :max_dlen]
        # x: (batch_size, days, max_daily_news, max_news_words, embedding_dim)

        x = self.embedding(x).float()

        mask = sequence_mask(news_len, max_len=max_nlen)
        mask = torch.unsqueeze(mask, dim=4)
        x *= mask
        # Word-level attention
        t = self.t(x)

        n = F.softmax(t, dim=3)*x
        # n: (batch_size, days, max_daily_news, embedding_dim)
        n = torch.sum(n, dim=3)

        mask = sequence_mask(day_len, max_len=max_dlen)
        mask = torch.unsqueeze(mask, dim=3)
        n *= mask

        # News-level attention
        u = self.u(n)
        d = F.softmax(u, dim=2) * n
        d = torch.sum(d, dim=2)
        # Sequence modeling
        h = self.gru(d)
        # Temporal attention
        o = self.o(h[0])
        v = F.softmax(o, dim=2) * h[0]
        v = torch.sum(v, dim=1)
        v.mean()
        # Discriminative Network (MLP)
        v = self.fc0(v)
        v = self.dropout(v) if training else v
        v = self.fc1(v)
        v = self.dropout(v) if training else v
        # v = v.view(x.size(0), -1)
        # v = self.layers(v)
        return self.fc_out(v)

class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()


    def forward(self, logits, labels, weights):
        w = torch.tensor(weights, dtype=torch.float32) * F.one_hot(labels, num_classes=3)
        w = torch.sum(w, dim=1)
        uw = F.cross_entropy(logits, labels)
        loss = torch.mean(w * uw)
        return loss

def compute_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=1)
    labels = labels.to(torch.int64)
    batch_size = int(logits.shape[0])
    return torch.sum(
        torch.eq(predictions, labels).to(torch.float32)) / batch_size

