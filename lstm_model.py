#define the nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

class LSTM_AE_Model(nn.Module):
    def __init__(self, input_size=10, hidden_dim=64):
        super(LSTM_Base_Model, self).__init__()
        self._hidden_dim = hidden_dim
        self._input_size = input_size
        self._lstm_encoder = nn.LSTM(self._input_size, self._hidden_dim, 1)
        self._lstm_decoder = nn.LSTM(self._hidden_dim, self._input_size, 1)
        self._hidden = None
        
    def forward(self, x):
        x, hidden = self._lstm_encoder(x)
        x, output = self._lstm_decoder(hidden)
        return output

    def train_model(self, dataset_generator, device, optimizer, criterion, epoch):
        for i in range(epoch):
            current_loss = 0.0
            num_examples = 0.0
            for data in dataset_generator:
                # get the inputs; data is a list of [inputs, labels]
                # zero the parameter gradients
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
            logging.info('[%d] epoch loss: %.3f' %
                        (epoch + 1, current_loss / num_examples))
        return current_loss / num_examples