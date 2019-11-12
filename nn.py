import torch.nn as nn
import torch.nn.functional as F 
import torch
import numpy as np


def num_flat_features(x):
    # (32L, 50L, 11L, 14L), 32 is batch_size
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    # print("after flatten shape: ",num_features)    
    return num_features


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer1_branch1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.layer1_branch2 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=51, stride=5, padding=25)
        self.layer1_branch3 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=101, stride=10, padding=50)

        self.bn1_branch1 = nn.BatchNorm1d(32)
        self.bn1_branch2 = nn.BatchNorm1d(32)
        self.bn1_branch3 = nn.BatchNorm1d(32)

        self.layer2_branch1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.layer2_branch2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.layer2_branch3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)

        self.bn2_branch1 = nn.BatchNorm1d(32)
        self.bn2_branch2 = nn.BatchNorm1d(32)
        self.bn2_branch3 = nn.BatchNorm1d(32)

        self.pool2_branch1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_branch2 = nn.MaxPool1d(kernel_size=30, stride=30)
        self.pool2_branch3 = nn.MaxPool1d(kernel_size=15, stride=15)
        
        self.layer3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 11), stride=(3, 11))

        self.layer4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.layer5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.layer6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # self.fc1 = nn.Linear(6144, 1024)
        self.fc1 = nn.Linear(1280, 256)
        # self.fc2 = nn.Linear(1024, 6)
        self.fc2 = nn.Linear(256, 6)
        # self.fc3 = nn.Linear(4096, 50)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = F.ReLU()
        
    def forward(self, x):
        # print x.size()
        # input: (batchSize, 1L, 80000L)
        # print("inside model")
        # print x.size()
        # x1 = self.relu(self.bn1_branch1(self.layer1_branch1(x))) 
        # x2 = self.relu(self.bn1_branch2(self.layer1_branch2(x)))
        # print x2.size()
        x3 = self.bn1_branch3(self.relu(self.layer1_branch3(x)))
        #print("layer 1 completed")
        # x1 = self.relu(self.bn2_branch1(self.layer2_branch1(x1)))
        # x2 = self.relu(self.bn2_branch2(self.layer2_branch2(x2)))
        # print x3.size()
        x3 = self.bn2_branch3(self.relu(self.layer2_branch3(x3)))

        # x1 = self.pool2_branch1(x1)
        # x2 = self.pool2_branch2(x2)
        x3 = self.pool2_branch3(x3)  
        # print x3.size()
        # x1 = torch.unsqueeze(x1, 1)
        # x2 = torch.unsqueeze(x2, 1)
        x3 = torch.unsqueeze(x3, 1)
        # print x3.size()
        # h = x2.clone().detach()
        h = x3.clone().detach()
        # print h.size()
        # h = torch.tensor(x2)    
        # print ("After Concatination: ", h.size())

        ##############  multiFeature formed above  ##############################

        h = x3
        h = self.layer3(h)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)  
        # print ("Layer 3: ", h.size())

        h = self.layer4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)  
        # print ("Layer 4: ", h.size())

        h = self.layer5(h)
        h = self.bn5(h)
        h = self.relu(h)
        h = self.pool5(h)
        # print ("Layer 5: ", h.size())

        h = self.layer6(h)
        h = self.bn6(h)
        h = self.relu(h)
        h = self.pool6(h)  
        # print ("Layer 6: ", h.size())

        h = h.view(-1, num_flat_features(h))
        # print h.size()
        h = self.fc1(h)
        # print h.size()
        h = F.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        # print ("Layer last: ", h.size())
        return h

mynet = MyNet().double()