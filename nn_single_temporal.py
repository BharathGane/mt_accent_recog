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
        super(MyNet, self).__init__()
        self.layer1_branch1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.layer1_branch2 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=51, stride=5, padding=25)
        self.layer1_branch3 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=101, stride=10, padding=50)
        self.layer1_branch4 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=151, stride=15, padding=75)
        self.layer1_branch5 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=201, stride=20, padding=100)
        self.layer1_branch6 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=251, stride=25, padding=125)
        self.layer1_branch7 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=301, stride=30, padding=150)
        self.layer1_branch8 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=501, stride=50, padding=250)
        self.layer1_branch9 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=601, stride=60, padding=300)
        self.layer1_branch10 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=751, stride=75, padding=375)
        self.layer1_branch11 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1000, stride=100, padding=500)
        self.layer1_branch12 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1501, stride=150, padding=750)

        self.bn1_branch1 = nn.BatchNorm1d(32)
        self.bn1_branch2 = nn.BatchNorm1d(32)
        self.bn1_branch3 = nn.BatchNorm1d(32)
        self.bn1_branch4 = nn.BatchNorm1d(32)
        self.bn1_branch5 = nn.BatchNorm1d(32)
        self.bn1_branch6 = nn.BatchNorm1d(32)
        self.bn1_branch7 = nn.BatchNorm1d(32)
        self.bn1_branch8 = nn.BatchNorm1d(32)
        self.bn1_branch9 = nn.BatchNorm1d(32)
        self.bn1_branch10 = nn.BatchNorm1d(32)
        self.bn1_branch11 = nn.BatchNorm1d(32)
        self.bn1_branch12 = nn.BatchNorm1d(32)

        self.layer2_branch1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.layer2_branch2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.layer2_branch3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.layer2_branch4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.layer2_branch5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.layer2_branch6 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.layer2_branch7 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.layer2_branch8 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.layer2_branch9 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.layer2_branch10 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.layer2_branch11 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.layer2_branch12 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)

        self.bn2_branch1 = nn.BatchNorm1d(32)
        self.bn2_branch2 = nn.BatchNorm1d(32)
        self.bn2_branch3 = nn.BatchNorm1d(32)
        self.bn2_branch4 = nn.BatchNorm1d(32)
        self.bn2_branch5 = nn.BatchNorm1d(32)
        self.bn2_branch6 = nn.BatchNorm1d(32)
        self.bn2_branch7 = nn.BatchNorm1d(32)
        self.bn2_branch8 = nn.BatchNorm1d(32)
        self.bn2_branch9 = nn.BatchNorm1d(32)
        self.bn2_branch10 = nn.BatchNorm1d(32)
        self.bn2_branch11 = nn.BatchNorm1d(32)
        self.bn2_branch12 = nn.BatchNorm1d(32)

        self.pool2_branch1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_branch2 = nn.MaxPool1d(kernel_size=30, stride=30)
        self.pool2_branch3 = nn.MaxPool1d(kernel_size=15, stride=15)
        self.pool2_branch4 = nn.MaxPool1d(kernel_size=10, stride=10)
        self.pool2_branch5 = nn.MaxPool1d(kernel_size=15, stride=15)
        self.pool2_branch6 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.pool2_branch7 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.pool2_branch8 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool2_branch9 = nn.MaxPool1d(kernel_size=15, stride=15)
        self.pool2_branch10 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2_branch11 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.pool2_branch12 = nn.MaxPool1d(kernel_size=1, stride=1)
        
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
        # self.fc1 = nn.Linear(5120, 1028)
        # self.fc2 = nn.Linear(1024, 6)
        self.fc2 = nn.Linear(1280, 256)
        self.fc3 = nn.Linear(256, 6)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 5 9 11 check out if it'll work
        # x1 = self.relu(self.bn1_branch1(self.layer1_branch1(x)))
        x2 = self.relu(self.bn1_branch2(self.layer1_branch2(x)))
        # x3 = self.relu(self.bn1_branch3(self.layer1_branch3(x)))
        # x4 = self.relu(self.bn1_branch4(self.layer1_branch4(x)))
        # x5 = self.relu(self.bn1_branch5(self.layer1_branch5(x)))
        # x6 = self.relu(self.bn1_branch6(self.layer1_branch6(x)))
        # x7 = self.relu(self.bn1_branch7(self.layer1_branch7(x)))
        # x8 = self.relu(self.bn1_branch8(self.layer1_branch8(x)))
        # x9 = self.relu(self.bn1_branch9(self.layer1_branch9(x)))
        # x10 = self.relu(self.bn1_branch10(self.layer1_branch10(x)))
        # x11 = self.relu(self.bn1_branch11(self.layer1_branch11(x)))
        # x12 = self.relu(self.bn1_branch12(self.layer1_branch12(x)))
        #print("layer 1 completed")
        # x1 = self.relu(self.bn2_branch1(self.layer2_branch1(x1)))
        x2 = self.relu(self.bn2_branch2(self.layer2_branch2(x2)))
        # x3 = self.relu(self.bn2_branch3(self.layer2_branch3(x3)))
        # x4 = self.relu(self.bn2_branch4(self.layer2_branch4(x4)))
        # x5 = self.relu(self.bn2_branch5(self.layer2_branch5(x5)))
        # x6 = self.relu(self.bn2_branch6(self.layer2_branch6(x6)))
        # x7 = self.relu(self.bn2_branch7(self.layer2_branch7(x7)))
        # x8 = self.relu(self.bn2_branch8(self.layer2_branch8(x8)))
        # x9 = self.relu(self.bn2_branch9(self.layer2_branch9(x9)))
        # x10 = self.relu(self.bn2_branch10(self.layer2_branch10(x10)))
        # x11 = self.relu(self.bn2_branch11(self.layer2_branch11(x11)))
        # x12 = self.relu(self.bn2_branch12(self.layer2_branch12(x12)))

        # x1 = self.pool2_branch1(x1)
        x2 = self.pool2_branch2(x2)
        # x3 = self.pool2_branch3(x3)
        # x4 = self.pool2_branch4(x4)
        # x5 = self.pool2_branch5(x5)
        # x6 = self.pool2_branch6(x6)
        # x7 = self.pool2_branch7(x7)
        # x8 = self.pool2_branch8(x8)
        # x9 = self.pool2_branch9(x9)
        # x10 = self.pool2_branch10(x10)  
        # x11 = self.pool2_branch11(x11)  
        # x12 = self.pool2_branch12(x12)  

        # x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        # x3 = torch.unsqueeze(x3, 1)  
        # x4 = torch.unsqueeze(x4, 1)  
        # x5 = torch.unsqueeze(x5, 1)
        # x6 = torch.unsqueeze(x6, 1)
        # x7 = torch.unsqueeze(x7, 1)
        # x8 = torch.unsqueeze(x8, 1)
        # x9 = torch.unsqueeze(x9, 1)
        # x10 = torch.unsqueeze(x10, 1)
        # x11 = torch.unsqueeze(x11, 1)
        # x12 = torch.unsqueeze(x12, 1)


        h = x2
        
        ##############  multiFeature formed above  ##############################
        
        
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
        # h = F.relu(self.fc1(h))
        # h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        h = self.fc3(h)
        # print ("Layer last: ", h.size())
        return h


mynet = MyNet().double()