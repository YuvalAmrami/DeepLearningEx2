#define the nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

class LSTM_AE_Classification_Model(nn.Module):
    def __init__(self, device, predicted_clasess, input_size=10, encoder_hidden_dim=64, decoder_hidden_dim=64):
        super(LSTM_AE_Classification_Model, self).__init__()
        self._encoder_hidden_dim = encoder_hidden_dim
        self._decoder_hidden_dim = decoder_hidden_dim
        self._input_size = input_size
        self._lstm_encoder = nn.LSTM(self._input_size, self._encoder_hidden_dim, 1)
        self._lstm_decoder = nn.LSTM(self._encoder_hidden_dim, self._decoder_hidden_dim, 1)
        self._device = device
        self._classification_layer = nn.Linear(self._decoder_hidden_dim, predicted_clasess)
        self._last_layer = nn.Linear(self._decoder_hidden_dim, self._input_size)
        
    def forward(self, x):
        seq_len = x.shape[0]
        _, (hidden, _) = self._lstm_encoder(x)
        hidden = hidden.repeat(seq_len, 1, 1)
        x, (decoder_last_hidden, _) = self._lstm_decoder(hidden)
        return F.relu(self._last_layer(x)), self._classification_layer(decoder_last_hidden.view(-1, self._decoder_hidden_dim))

    def train_model(self, dataset_generator, device, optimizer, criterion, classification_criterion, clip, sample_size, seq_len):
        self.train(True)
        current_loss = 0.0
        accuracy = 0.0
        num_labels = 0.0
        num_batches = 0.0
        for data in dataset_generator:
            # get the inputs; data is a list of [inputs, labels]
            # zero the parameter gradients
            inputs, labels = data
            inputs = inputs.view(seq_len, -1, sample_size)
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            # forward + backward + optimize
            outputs, predictions = self(inputs)
            loss = criterion(outputs, inputs)
            loss += classification_criterion(predictions, labels)
            # print statistics
            current_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            accuracy += (labels == torch.argmax(predictions, dim=1)).float().sum()
            num_labels += labels.shape[0]
            num_batches += 1
        return current_loss / num_batches, accuracy / num_labels

    def evaluate_model(self, dataset_generator, device, criterion, classification_criterion, sample_size, seq_len):
        self.eval()
        current_loss = 0.0
        accuracy = 0.0
        num_labels = 0.0
        num_batches = 0.0
        with torch.no_grad():
            for data in dataset_generator:
                # get the inputs; data is a list of [inputs, labels]
                # zero the parameter gradients
                inputs, labels = data
                inputs = inputs.view(seq_len, -1, sample_size)
                inputs, labels = inputs.to(device), labels.to(device)
                # forward + backward + optimize
                outputs, predictions = self(inputs)
                loss = criterion(outputs, inputs)
                loss += classification_criterion(predictions, labels)
                # print statistics
                current_loss += loss.item()
                accuracy += (labels == torch.argmax(predictions, dim=1)).float().sum()
                num_labels += labels.shape[0]
                num_batches += 1
        return current_loss / num_batches, accuracy / num_labels