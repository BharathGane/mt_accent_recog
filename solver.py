from torch import optim
import utils
import gc
import torch
from nn import mynet
from nn import MyNet
from torch import nn
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
print(device)
model = MyNet()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay= 0.0005)
exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 140], gamma=0.1)
criterion = nn.CrossEntropyLoss().cuda()
file_name_label = {"ABA":"Arabic","SKA":"Arabic","YBAA":"Arabic","ZHAA":"Arabic","BWC":"Chinese",
                "BWC":"Chinese","LXC":"Chinese","NCC":"Chinese","TXHC":"Chinese",
                "ASI":"Hindi","RRBI":"Hindi","SVBI":"Hindi","TNI":"Hindi",
                "HJK":"Korean","HKK":"Korean","YDCK":"Korean","YKWK":"Korean",
                "EBVS":"Spanish","ERMS":"Spanish","MBMPS":"Spanish","NJS":"Spanish",
                "HQTV":"Vietnamese","PNV":"Vietnamese","THV":"Vietnamese","TLV":"Vietnamese"}
label_file_name = {'Chinese': ['TXHC', 'BWC', 'LXC', 'NCC'], 
                'Vietnamese': ['HQTV', 'TLV', 'THV', 'PNV'], 
                'Hindi': ['ASI', 'RRBI', 'TNI', 'SVBI'], 
                'Spanish': ['ERMS', 'NJS', 'EBVS', 'MBMPS'], 
                'Korean': ['YKWK', 'HKK', 'YDCK', 'HJK'], 
                'Arabic': ['YBAA', 'SKA', 'ZHAA', 'ABA']}
labels = ["Chinese","Vietnamese","Hindi","Spanish","Korean","Arabic"]

def data_loader(value):
    freq = 41000
    chunk_freq = 61500
    time_each_chunk = float(chunk_freq)/float(freq)
    traning_time_in_sec = 2
    number_of_chunks = int(traning_time_in_sec/time_each_chunk)
    print freq,chunk_freq,time_each_chunk,traning_time_in_sec,number_of_chunks
    if value == "train":
        for i in labels:
            for j in range(len(label_file_name[i])-1):
                # data_array = utils.read_audio_file_data(os.path.join("./combined_wav_files",label_file_name[i][j]+".wav"))
                source = os.path.join("./combined_wav_files",label_file_name[i][j]+".wav")
                # for k in range(0,traning_time_in_sec/time_each_chunk):
                for k in utils.read_audio_file_data_chunks(source,chunk_freq,number_of_chunks):
                    gc.collect()
                    yield (torch.tensor(np.tile(k,(32,1,1)),dtype = torch.float),torch.tensor(np.tile(np.asarray(labels.index(i)),(32)),dtype = torch.long).cuda())
    elif value == "test":
        for i in labels:
            source = os.path.join("./combined_wav_files",label_file_name[i][3]+".wav")
            # data_array = utils.read_audio_file_data(os.path.join("./combined_wav_files",label_file_name[i][3]+".wav"))
            for k in utils.read_audio_file_data_chunks(source,chunk_freq,number_of_chunks):
            # for k in range(0,traning_time_in_sec/time_each_chunk):
                gc.collect()
                # yield (torch.from_numpy(np.tile(data_array[k*chunk_freq:k*chunk_freq+chunk_freq],(32,1,1)),dtype = torch.cuda.DoubleTensor).cuda(),torch.from_numpy(np.tile(np.asarray(labels.index(i)),(32)),dtype = torch.cuda.LongTensor).cuda())
                yield (torch.tensor(np.tile(k,(32,1,1)),dtype = torch.float),torch.tensor(np.tile(np.asarray(labels.index(i)),(32)),dtype = torch.long).cuda())

def test():
    model.load_state_dict(torch.load('./model.pt'))
    model.eval()
    class_correct = list(0. for i in range(6))
    class_total = list(0. for i in range(6))
    for i, data in enumerate(data_loader("test"), 0):
        gc.collect()
        inputs, labels = data
        outputs = model(inputs).cuda()
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        # print c,predicted
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
            # print class_total,class_correct
    return class_total,class_correct

def train():
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(data_loader("train"), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # print labels
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs).cuda()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            gc.collect()
    torch.save(model.state_dict(), "./model.pt")
    # print('Finished Training')
