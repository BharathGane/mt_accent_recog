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
model = MyNet().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay= 0.0005)
exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 140], gamma=0.1)
criterion = nn.CrossEntropyLoss()
file_name_label = {"ABA":"Arabic","SKA":"Arabic","YBAA":"Arabic","ZHAA":"Arabic","BWC":"Chinese",
                "BWC":"Chinese","LXC":"Chinese","NCC":"Chinese","TXHC":"Chinese",
                "ASI":"Hindi","RRBI":"Hindi","SVBI":"Hindi","TNI":"Hindi",
                "HJK":"Korean","HKK":"Korean","YDCK":"Korean","YKWK":"Korean",
                "EBVS":"Spanish","ERMS":"Spanish","MBMPS":"Spanish","NJS":"Spanish",
                "HQTV":"Vietnamese","PNV":"Vietnamese","THV":"Vietnamese","TLP":"Vietnamese"}
label_file_name = {'Chinese': ['TXHC', 'BWC', 'LXC', 'NCC'], 
                'Vietnamese': ['HQTV', 'TLP', 'THV', 'PNV'], 
                'Hindi': ['ASI', 'RRBI', 'TNI', 'SVBI'], 
                'Spanish': ['ERMS', 'NJS', 'EBVS', 'MBMPS'], 
                'Korean': ['YKWK', 'HKK', 'YDCK', 'HJK'], 
                'Arabic': ['YBAA', 'SKA', 'ZHAA', 'ABA']}
labels = label_file_name.keys()

def data_loader(value):
    if value == "train":
        for i in labels:
            gc.collect()
            data_array = utils.read_audio_file_data(os.path.join("./combined_wav_files",label_file_name[i][0]+".wav"))
            for j in range(0,500):
                gc.collect()
                yield (torch.from_numpy(np.tile(data_array[j*80000:j*80000+80000],(32,1,1))),i)
    elif value == "test":
        for i in labels:
            gc.collect()
            data_array = utils.read_audio_file_data(os.path.join("./combined_wav_files",label_file_name[i][1]+".wav"))
            for j in range(0,1):
                gc.collect()
                yield (torch.from_numpy(np.tile(data_array[j*80000:j*80000+80000]),(32,1,1)),i)
def test():
    for i, data in enumerate(data_loader("test"), 0):
        inputs, labels = data
        outputs = mynet(inputs)
        print outputs,labels

def train():
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(data_loader("train"), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = mynet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
