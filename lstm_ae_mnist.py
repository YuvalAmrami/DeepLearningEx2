import argparse
import dataset_utils
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from lstm_model import LSTM_AE_Model
from lstm_model_with_classifcation import LSTM_AE_Classification_Model
import logging
from torch.utils.data import DataLoader
import pickle
import os
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='train a lstm auto encoder over synthetic dataset')
parser.add_argument('--epochs', type=int, help='number of ephocs')
parser.add_argument('--optimizer', help='optimizer for algorithm (Adam,SGD)')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--gd_clip', type=float, help='gradient clipping value')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--hidden_dim', type=int, help='hidden dim')
parser.add_argument('--prediction', type=bool, default=False, help='output file for statistics')
parser.add_argument('--stats_file_prefix', help='output folder for tensorboard')
parser.add_argument('--model_file_prefix', help='output file for model')
parser.add_argument('--input_size', type=int, help='input size')


args = parser.parse_args()

epochs = args.epochs
optimizer = args.optimizer
learning_rate = args.lr
clip = args.gd_clip
batch_size = args.batch_size
hidden_dim = args.hidden_dim
is_prediction_task = args.prediction
save_path = args.model_file_prefix
save_path = '{}_{}_{}_{}.model'.format(save_path, learning_rate, hidden_dim, clip)
input_size = args.input_size
stats_file = args.stats_file_prefix
print(is_prediction_task)

logging.basicConfig(filename="lstm_ae_train.log", filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seq_len = int(784 / input_size)

if is_prediction_task:
    model = LSTM_AE_Classification_Model(device, 10, input_size=input_size, hidden_dim=hidden_dim, seq_len=784)
    model.to(device)
    classification_criterion = nn.CrossEntropyLoss()
    train, val, test = dataset_utils.load_mnist_dataset(True)
else:
    model = LSTM_AE_Model(device, input_size=input_size, hidden_dim=hidden_dim, seq_len=784)
    model.to(device)
    train, val, test = dataset_utils.load_mnist_dataset(False)


dataset_train = DataLoader(train, batch_size=args.batch_size, shuffle=True)
dataset_validation = DataLoader(val, batch_size=args.batch_size, shuffle=True)
dataset_test = DataLoader(test, batch_size=args.batch_size, shuffle=True)


criterion = nn.MSELoss()
if optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#start training
logging.info('start training')
best_loss = 99999
best_accuracy = 0.0

comment = f'gradient_clip = {clip} lr = {learning_rate} hidden_dim = {hidden_dim} ephochs = {epochs}'
tb = SummaryWriter(log_dir=os.path.join(stats_file, comment))
for epoch in range(epochs):  # loop over the dataset multiple times
    if is_prediction_task:
        train_reconstruction_loss, train_prediction_loss, train_acc = model.train_model(dataset_train, device, optimizer, criterion, classification_criterion, clip, input_size, seq_len)
        val_reconstruction_loss, val_prediction_loss, val_acc = model.evaluate_model(dataset_validation, device, criterion, classification_criterion, input_size, seq_len)
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), save_path)
        tb.add_scalar("Train Reconstruction Loss", train_reconstruction_loss, epoch)
        tb.add_scalar("Validation Reconstruction Loss", val_reconstruction_loss, epoch)
        tb.add_scalar("Train Prediction Loss", train_prediction_loss, epoch)
        tb.add_scalar("Validation Prediction Loss", val_prediction_loss, epoch)
        tb.add_scalar("Train Accuracy", train_acc, epoch)
        tb.add_scalar("Validation Accuracy", val_acc, epoch)
        logging.info('finished epoch, best accuracy {}'.format(epoch, best_accuracy))
    else:
        train_loss = model.train_model(dataset_train, device, optimizer, criterion, clip, input_size, seq_len)
        val_loss = model.evaluate_model(dataset_validation, device, criterion, input_size, seq_len)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
        tb.add_scalar("Train Loss", train_loss, epoch)
        tb.add_scalar("Validation Loss", val_loss, epoch)
        logging.info('finished epoch {}, best loss {}'.format(epoch, best_loss))

model.load_state_dict(torch.load(save_path))
if is_prediction_task:
    test_loss = model.evaluate_model(dataset_test, device, criterion, classification_criterion, input_size, seq_len)
else:    
    test_loss = model.evaluate_model(dataset_test, device, criterion, input_size, seq_len)
logging.info('test loss {}'.format(test_loss))

print('Finished Training')

