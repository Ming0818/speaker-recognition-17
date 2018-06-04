#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

"""class BinaryModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryModel, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc3 = nn.Linear(hidden_size, 1)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        hid = self.fc1(input)
        hid2 = self.fc2(hid)
        hid3 = self.fc3(hid2)
        output = self.softmax(hid3)
        return output"""
    
class BinaryModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryModel, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(hidden_size, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y