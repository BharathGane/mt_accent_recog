from torch import optim
import utils
import gc
import torch
from nn_single_temporal import MyNet
from torch import nn
import os
import numpy as np
import pickle
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
print(device)
model = MyNet().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay= 0.005)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=map(lambda x: x*5,range(100)), gamma=0.1)
criterion = nn.CrossEntropyLoss().to(device)
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
labels_id = {}
labels_index = [map(float,(range(6)))]
for i in range(len(labels)):
    labels_id[labels[i]] = list(0 for i in range(6))
    labels_id[labels[i]][i] = 1
final_data = []

for i in labels:
    final = []
    file_indexes = range(len(label_file_name[i]))
    for j in file_indexes:
        source = os.path.join("./pkl_mini/",label_file_name[i][j]+".pkl")
        with open(source) as file_:
            data = pickle.load(file_)
        final.append(data)
    final_data.append(final)
final_data = torch.tensor(np.asarray(final_data)).to(device)
gc.collect()


def data_loader(value):
    freq = 44100
    chunk_freq = 66150
    time_each_chunk = float(chunk_freq)/float(freq)
    traning_time_in_sec = 750
    number_of_chunks = int(traning_time_in_sec/time_each_chunk)
    # if value == "train":
    #     # for iterator in range(number_of_chunks):
    #     for i in labels:
    #         file_indexes = range(len(label_file_name[i])-1)
    #         for j in file_indexes:
    #             source = os.path.join("./pkl_mini/",label_file_name[i][j]+".pkl")
    #             for k in utils.read_audio_dump(source,chunk_freq,number_of_chunks):
    #                 gc.collect()
    #                 # yield (torch.tensor(np.tile(k,(1,1,1)),dtype = torch.float).cuda(),torch.tensor(np.tile(np.asarray(labels_index),(1)),dtype = torch.float).cuda())
    #                 yield (torch.tensor(np.tile(k,(1,1,1)),dtype = torch.float).cuda(),torch.tensor(np.tile(np.asarray(labels.index(i)),(1)),dtype = torch.float).cuda())
    if value == "train":
        for iterator in range(number_of_chunks):
            for i in labels:
                file_indexes = range(len(label_file_name[i])-1)
                for j in file_indexes:
                    data = final_data[labels.index(i)][j][iterator*chunk_freq:iterator*chunk_freq+chunk_freq]
                    yield (data.repeat((1,1,1)).cuda(),torch.tensor(np.tile(np.asarray(labels.index(i)),(1)),dtype = torch.long).cuda())

    elif value == "test":
        for i in labels:
            for j in [3]:
                for iterator in range(number_of_chunks):
                    data = final_data[labels.index(i)][j][iterator*chunk_freq:iterator*chunk_freq+chunk_freq]
                    yield (data.repeat((1,1,1)).cuda(),torch.tensor(np.tile(np.asarray(labels.index(i)),(1)),dtype = torch.long).cuda())

def test():
    model.load_state_dict(torch.load('./model11.pt'))
    model.eval()
    class_correct = list(0. for i in range(6))
    class_total = list(0. for i in range(6))
    for i, data in enumerate(data_loader("test"), 0):
        gc.collect()
        inputs, labels = data[0].to(device),data[1].to(device)
        outputs = model(inputs).to(device)
        _, predicted = torch.max(outputs, 1)
        # print predicted
        # print labels
        if predicted == labels:
            class_correct[labels[0]] += 1
        class_total[labels[0]] += 1
        # print class_total,class_correct
    # print outputs
    print sum(class_correct)/sum(class_total)
    return class_total,class_correct

def train():
    # model.load_state_dict(torch.load('./model2.pt'))
    # model.eval()
    for epoch in range(50):  # loop over the dataset multiple times
        print optimizer
        running_loss = 0.0
        print "traning started for ",str(epoch),"epoch"
        for i, data in enumerate(data_loader("train"), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device),data[1].to(device)
            # print labels
            # zero the parameter gradients
            optimizer.zero_grad()
            model.zero_grad()
            model.train()
            # forward + backward + optimize
            outputs = model(inputs).to(device)
            # print outputs, labels
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print loss.item()
            if i % 100  == 99:    # print every 2000 mini-batches
                print [epoch + 1, i + 1, running_loss/99]
                running_loss = 0.0
            gc.collect()
        torch.save(model.state_dict(), "./model11.pt")
        print "traning ended for ",str(epoch),"epoch"
        print "testing started after ",str(epoch),"epoch"
        print test()
        print "testing ended after",str(epoch),"epoch"
    print('Finished Training')
