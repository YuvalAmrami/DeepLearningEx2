#define the nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

class LSTM_AE_Model(nn.Module):
    def __init__(self, device, input_size=10, hidden_dim=64):
        super(LSTM_AE_Model, self).__init__()
        self._hidden_dim = hidden_dim
        self._input_size = input_size
        self._lstm_encoder = nn.LSTM(self._input_size, self._hidden_dim, 1)
        self._lstm_decoder = nn.LSTM(self._hidden_dim, self._input_size, 1)
        self._decoder_hidden = None
        self._device = device

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self._input_size).to(self._device)
        cell = torch.zeros(1, batch_size, self._input_size).to(self._device)
        self._decoder_hidden = hidden, cell
        
    def forward(self, x):
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        x, (hidden, cell) = self._lstm_encoder(x)
        output = torch.FloatTensor(seq_len, batch_size, 1).to(self._device)
        self.init_hidden(batch_size)
        for i in range(seq_len):
            x, (self._decoder_hidden) = self._lstm_decoder(hidden, self._decoder_hidden)
            output[i, :, :] = self._decoder_hidden[0]
        return output

    def train_model(self, dataset_generator, device, optimizer, criterion, sample_size):
        self.train(True)
        current_loss = 0.0
        num_examples = 0.0
        for data in dataset_generator:
            # get the inputs; data is a list of [inputs, labels]
            # zero the parameter gradients
            data = data.view(sample_size, -1, 1)
            optimizer.zero_grad()
            inputs = data.to(device)
            # forward + backward + optimize
            outputs = self(inputs)

            loss = criterion(outputs, inputs)
            # print statistics
            current_loss += loss.item()
            loss.backward()
            optimizer.step()
            num_examples += 1
        return current_loss / num_examples

    def evaluate_model(self, dataset_generator, device, criterion, sample_size):
        self.eval()
        current_loss = 0.0
        num_examples = 0.0
        with torch.no_grad():
            for data in dataset_generator:
                # get the inputs; data is a list of [inputs, labels]
                # zero the parameter gradients
                data = data.view(sample_size, -1, 1)
                inputs = data.to(device)
                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                # print statistics
                current_loss += loss.item()
                num_examples += 1
        return current_loss / num_examples