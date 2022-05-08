import numpy as np
from torch.utils import data
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()


    def forward(self, logits, labels, weights):
        w = torch.tensor(weights, dtype=torch.float32) * F.one_hot(labels, num_classes=3)
        w = torch.sum(w, dim=1)
        uw = F.cross_entropy(logits, labels)
        loss = torch.mean(w * uw)
        return loss

class getdataset(data.Dataset):
    def __init__(self, data, label):
        self.data = np.asarray(data)
        self.label = np.asarray(label)

    def __getitem__(self, item):

        x, date_len,  news_len = dataset._get_idxes_len(self.data[item], max_data_len, max_news_len)
        label = torch.tensor(self.label[item])
        return torch.from_numpy(x), torch.tensor(date_len), torch.tensor(news_len), label

    def __len__(self):
        return len(self.data)
def compute_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=1)
    labels = labels.to(torch.int64)
    batch_size = int(logits.shape[0])
    return torch.sum(
        torch.eq(predictions, labels).to(torch.float32)) / batch_size

import torch.optim as optim
def training(batch_size, n_epoch, lr, train, valid, model, device):
    t_loss_list = []
    t_acc_list = []
    v_loss_list = []
    v_acc_list = []
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train()
    # 定义损失函数
    t_batch = len(train)
    v_batch = len(valid)
    criterion = Myloss()
   # criterion()
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        for i, (days, day_lens, news_lens, labels) in enumerate(train):
            days = days.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(days, day_lens, news_lens)
            loss = criterion(outputs, labels, weights)
            # x = myloss()
            # a = x(outputs, labels, weights)
            loss.backward()
            optimizer.step()
            correct = compute_accuracy(outputs, labels.to(torch.double))
            total_acc += correct
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
                epoch + 1, i + 1, t_batch, loss.item(), correct * 100), end='\r')
        t_loss_list.append(total_loss / t_batch)
        t_acc_list.append(float(total_acc) / t_batch * 100)
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss / t_batch, total_acc / t_batch * 100))

        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (days, day_lens, news_lens, labels) in enumerate(valid):
                days = days.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.long)
                outputs = model(days, day_lens, news_lens, training=False)
                loss = criterion(outputs, labels, weights)
                correct = compute_accuracy(outputs, labels.to(torch.double))
                total_acc += correct
                total_loss += loss.item()
            v_loss_list.append(total_loss / v_batch)
            v_acc_list.append(float(total_acc) / v_batch * 100)
            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss / v_batch, total_acc / v_batch * 100))
            print('-----------------------------------------------')
            model.train()

    return t_loss_list, t_acc_list, v_loss_list, v_acc_list
def testing(batch_size, test, model, device):
    model.eval()
    t_batch = len(test)
    ret_output = []
    criterion = Myloss()
    total_loss, total_acc, best_acc = 0, 0, 0
    with torch.no_grad():
        for i, (days, day_lens, news_lens, labels) in enumerate(test):
            days = days.to(device)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(days, day_lens, news_lens)
            correct = compute_accuracy(outputs, labels.to(torch.double))

            loss = criterion(outputs, labels, weights)
            total_loss += loss.item()
            total_acc += (correct / batch_size)
            outputs = torch.argmax(outputs, dim=1)
            ret_output.append(outputs)
            print('\nTest | Loss:{:.5f} Acc: {:.3f}'.format(total_loss / t_batch, total_acc / t_batch * 100))
    return ret_output

#main
dataset = Dataset()
dd = dataset.date_tweets
batch_size = 8
max_data_len = dataset.max_date_len
max_news_len = dataset.max_news_len
wordvec = dataset.word2vec
weights = dataset.class_weights
train_x, train_y, dev_x, dev_y, test_x, test_y = dataset.map_stocks_tweets()
train = getdataset(train_x, train_y)


dev = getdataset(dev_x, dev_y)
model = HAN(wordvec, 64, 8, 0.2)
#model.initialize()
device = torch.device('cpu')

dev_load = data.DataLoader(dev, batch_size)
train_load = data.DataLoader(train, batch_size)
x = training(batch_size, 1, 0.00005, train_load, dev_load, model, device)
print(x)
pre = testing(batch_size,dev_load, model, device)