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
        self._relative_seq_len = int(self._seq_len / self._input_size)

    def forward(self, x):
        out, (hidden, _) = self._lstm_encoder(x)
        hidden = hidden.view(-1, 1, self._hidden_dim)
        hidden = hidden.repeat(1, self._seq_len, 1)
        x, _ = self._lstm_decoder(hidden)
        x = x.reshape(-1, self._seq_len)
        x = self._last_layer(x)
        y = self.pred_linear_layer(out)
        return x.reshape(-1, self._relative_seq_len, self._input_size), y

    def train_model(self, dataset_generator, device, optimizer, criterion, clip, sample_size, seq_len):
        self.train(True)
        num_batches = 0.0
        current_reconstruction_loss = 0.0
        current_prediction_loss = 0.0
        for inputs in dataset_generator:
            # get the inputs; data is a list of [inputs, labels]
            # zero the parameter gradients
            x_in = inputs[:, :-1]
            y_in = inputs[:, 1:]
            x_in = x_in.view(-1, seq_len, sample_size)
            y_in = y_in.view(-1, seq_len, sample_size)
            optimizer.zero_grad()
            x_in = x_in.to(device)
            y_in = y_in.to(device)
            # forward + backward + optimize
            outputs, y_pred = self(x_in)
            reconstruction_loss = criterion(outputs, x_in)
            prediction_loss = criterion(y_in, y_pred)
            current_reconstruction_loss += reconstruction_loss.item()
            current_prediction_loss += prediction_loss.item()
            loss = reconstruction_loss + prediction_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            num_batches += 1
        return current_reconstruction_loss / num_batches, current_prediction_loss / num_batches

    def evaluate_model(self, dataset_generator, device, criterion, sample_size, seq_len):
        self.eval()
        current_reconstruction_loss = 0.0
        current_prediction_loss = 0.0
        num_batches = 0.0
        with torch.no_grad():
            for inputs in dataset_generator:
                # get the inputs; data is a list of [inputs, labels]
                x_in = inputs[:, :-1]
                y_in = inputs[:, 1:]
                
                x_in = x_in.view(-1, seq_len, sample_size)
                y_in = y_in.view(-1, seq_len, sample_size)
                x_in = x_in.to(device)
                y_in = y_in.to(device)

                outputs, y_pred = self(x_in)
                reconstruction_loss = criterion(outputs, x_in)
                prediction_loss = criterion(y_in, y_pred)
                current_reconstruction_loss += reconstruction_loss.item()
                current_prediction_loss += prediction_loss.item()
                num_batches += 1
        return current_reconstruction_loss / num_batches, current_prediction_loss / num_batches