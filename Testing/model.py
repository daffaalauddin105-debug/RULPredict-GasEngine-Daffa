# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:38:21 2026

@author: LOQ
"""
import torch.nn as nn

class reg_GRU(nn.Module):
    def __init__(self, input_size=10, hidden_size=50, output_size=1):
        super(reg_GRU, self).__init__()
        # input_size=1 karena kita hanya memonitor 1 variabel (sinus bernoise)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: [Batch, 10, 1]
        out, _ = self.gru(x)
        # out[:, -1, :] mengambil hidden state terakhir dari sequence
        return self.fc(out[:, -1, :]) # Hasil: [Batch, 5]

class BiLSTM(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])