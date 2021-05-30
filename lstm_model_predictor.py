#define the nn
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import logging

class LSTM_AE_Model_pred(nn.Module):
    def __init__(self, device, input_size=10, hidden_dim=64, seq_len=50):
        super(LSTM_AE_Model_pred, self).__init__()
        self._hidden_dim = hidden_dim
        self._seq_len = seq_len
        self._input_size = input_size
        self._lstm_encoder = nn.LSTM(self._input_size, self._hidden_dim, 1, batch_first=True)
        self._lstm_decoder = nn.LSTM(self._hidden_dim, self._input_size, 1, batch_first=True)
        self._device = device
        self._last_layer = nn.Linear(self._seq_len, self._seq_len)
        self.pred_linear_layer = nn.Linear(self._hidden_dim, self._input_size)

    def forward(self, x):
        # y = x[1:]
        out, (hidden, _) = self._lstm_encoder(x)
        hidden = hidden.view(-1, 1, self._hidden_dim)
        hidden = hidden.repeat(1, self._seq_len, 1)
        x, _ = self._lstm_decoder(hidden)
        x = x.view(-1, self._seq_len, self._input_size)
        if self._input_size == 1:
            x = torch.squeeze(x, 2)
        x = self._last_layer(x)
        y = self.pred_linear_layer(out[:,:,:])
        # if self._input_size == 1:
        #     y = torch.squeeze(y, 2)
        if self._input_size == 1:
            return x.unsqueeze(2), y
        return x, y

    def train_model(self, dataset_generator, device, optimizer, criterion, clip, sample_size, seq_len):
        self.train(True)
        current_loss = 0.0
        num_batches = 0.0
        for inputs in dataset_generator:
            # get the inputs; data is a list of [inputs, labels]
            # zero the parameter gradients
            y_in = inputs[:, 1:]
            x_in = inputs[:,:-1]
            inputs = x_in.view(-1, seq_len, sample_size)
            y_in = y_in.view(-1, seq_len, sample_size)
            optimizer.zero_grad()
            inputs = inputs.to(device)
            y_in = y_in.to(device)
            # forward + backward + optimize
            outputs, y_pred = self(inputs)
            loss1 = criterion(outputs, inputs)
            # print statistics
            # if self._input_size == 1:
            #     y_in = torch.squeeze(y_in, 2)
            loss2 = criterion(y_in, y_pred)
            loss = (loss1+loss2)/2
            current_loss += loss.item() + loss2.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            #torch.nn.utils.clip_grad_norm_(self._lstm_decoder.parameters(), clip)
            optimizer.step()
            num_batches += 1
        return current_loss / num_batches

    def evaluate_model(self, dataset_generator, device, criterion, sample_size, seq_len):
        self.eval()
        current_loss = 0.0
        num_batches = 0.0
        with torch.no_grad():
            for inputs in dataset_generator:
                # get the inputs; data is a list of [inputs, labels]
                # zero the parameter gradients

                y_in = inputs[:, 1:]
                x_in = inputs[:, :-1]

                x_in = x_in.view(-1, seq_len, sample_size)
                x_in = x_in.to(device)
                y_in = y_in.to(device)

                # forward + backward + optimize
                outputs, y_pred = self(x_in)
                loss = criterion(outputs, x_in)
                loss2 = criterion(y_in, y_pred)

                # print statistics
                current_loss += loss.item()+loss2.item()
                num_batches += 1
        return current_loss / num_batches