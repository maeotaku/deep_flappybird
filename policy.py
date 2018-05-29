import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    def __init__(self, number_of_frames):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(number_of_frames, 10, kernel_size=5)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3380, 512)
        self.fc1_drop = nn.Dropout()
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(512, 128)
        self.fc2_drop = nn.Dropout()
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(128, 2)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 3380)
        x = F.relu(self.fc1_drop(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2_drop(self.fc2(x)))
        x = self.fc3(x)
        #F.sigmoid(x)
        return F.softmax(x, dim=1)
