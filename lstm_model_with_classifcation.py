#define the nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

class LSTM_AE_Classification_Model(nn.Module):
    def __init__(self, device, predicted_clasess, input_size=10, hidden_dim=64, seq_len=784):
        super(LSTM_AE_Classification_Model, self).__init__()
        self._hidden_dim = hidden_dim
        self._input_size = input_size
        self._seq_len = seq_len
        self._lstm_encoder = nn.LSTM(self._input_size, self._hidden_dim, 1, batch_first=True)
        self._lstm_decoder = nn.LSTM(self._hidden_dim, self._input_size, 1, batch_first=True)
        self._device = device
        self._classification_layer = nn.Linear(self._input_size, predicted_clasess)
        self._last_layer = nn.Linear(self._seq_len, self._seq_len)
        self._relative_seq_len = int(self._seq_len / self._input_size)
        
    def forward(self, x):
        _, (hidden, _) = self._lstm_encoder(x)
        hidden = hidden.view(-1, 1, self._hidden_dim)
        hidden = hidden.repeat(1, self._relative_seq_len, 1)
        x, (decoder_last_hidden, _) = self._lstm_decoder(hidden)
        x = x.reshape(-1, self._seq_len)
        x = self._last_layer(x)
        return x.reshape(-1, self._relative_seq_len, self._input_size), self._classification_layer(decoder_last_hidden.view(-1, self._input_size))

    def train_model(self, dataset_generator, device, optimizer, criterion, classification_criterion, clip, sample_size, seq_len):
        self.train(True)
        current_reconstruction_loss = 0.0
        current_prediction_loss = 0.0
        accuracy = 0.0
        num_labels = 0.0
        num_batches = 0.0
        for data in dataset_generator:
            # get the inputs; data is a list of [inputs, labels]
            # zero the parameter gradients
            inputs, labels = data
            inputs = inputs.view(-1, seq_len, sample_size)
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            # forward + backward + optimize
            outputs, predictions = self(inputs)
            reconstruction_loss = criterion(outputs, inputs)
            prediction_loss = classification_criterion(predictions, labels)
            # print statistics
            current_reconstruction_loss += reconstruction_loss.item()
            current_prediction_loss += prediction_loss.item()
            loss = reconstruction_loss + prediction_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()
            accuracy += (labels == torch.argmax(predictions, dim=1)).float().sum()
            num_labels += labels.shape[0]
            num_batches += 1
        return current_reconstruction_loss / num_batches, current_prediction_loss / num_batches, accuracy / num_labels

    def evaluate_model(self, dataset_generator, device, criterion, classification_criterion, sample_size, seq_len):
        self.eval()
        current_reconstruction_loss = 0.0
        current_prediction_loss = 0.0
        accuracy = 0.0
        num_labels = 0.0
        num_batches = 0.0
        with torch.no_grad():
            for data in dataset_generator:
                # get the inputs; data is a list of [inputs, labels]
                # zero the parameter gradients
                inputs, labels = data
                inputs = inputs.view(-1, seq_len, sample_size)
                inputs, labels = inputs.to(device), labels.to(device)
                # forward + backward + optimize
                outputs, predictions = self(inputs)
                reconstruction_loss = criterion(outputs, inputs)
                prediction_loss = classification_criterion(predictions, labels)
                # print statistics
                current_reconstruction_loss += reconstruction_loss.item()
                current_prediction_loss += prediction_loss.item()
                accuracy += (labels == torch.argmax(predictions, dim=1)).float().sum()
                num_labels += labels.shape[0]
                num_batches += 1
        return current_reconstruction_loss / num_batches, current_prediction_loss / num_batches, accuracy / num_labels