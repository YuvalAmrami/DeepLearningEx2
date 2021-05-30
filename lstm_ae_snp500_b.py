import argparse

from torch.utils.tensorboard import SummaryWriter

import dataset_utils as DU
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from lstm_model import LSTM_AE_Model
from lstm_model_predictor import LSTM_AE_Model_pred
from lstm_model_with_classifcation import LSTM_AE_Classification_Model
import logging
from torch.utils.data import DataLoader
import pickle

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os



parser = argparse.ArgumentParser(description='train a lstm auto encoder over synthetic dataset')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
parser.add_argument('--optimizer', default='SGD', help='optimizer for algorithm (Adam,SGD)')
parser.add_argument('--lr', type=float, default=10**-3, help='learning rate')
parser.add_argument('--gd_clip', type=int, default=1, help='gradient clipping value')
parser.add_argument('--batch_size', type=int, default=500, help='batch size')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dim')
parser.add_argument('--prediction', type=bool, default=False, help='output file for statistics')
# parser.add_argument('--num_of_days_back', default=40, type=int, help='num of days to calculate back, the number of ts')
parser.add_argument('--stats_file_prefix',  default='stas_snp_0', help='output file for statistics')
parser.add_argument('--model_file_prefix',  default='mod_snp_0', help='output file for model')
parser.add_argument('--input_size', default=1, type=int, help='input size')
parser.add_argument('--attempt_name', default='snp_0', help='name for train, validation and test files of this attempt')

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
attempt_name = args.attempt_name
# num_of_days = args.num_of_days_back
# stock_name = args.stock_name

print(is_prediction_task)

stocks = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')
# googl = stocks.loc[stocks['symbol'] == 'GOOGL']
# amzn = stocks.loc[stocks['symbol'] == 'AMZN']
# stock = stocks.loc[stocks['symbol'] == stock_name]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()



if optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

test, rest = DU.split_stocks_dataset_old(stocks, attempt_name)

dataset_test = DataLoader(test, batch_size=args.batch_size, shuffle=True)


for index in range(len(rest)):
    train_list =[]
    val =rest[index]
    for index2 in range(len(rest)):
        if (index!=index2):
            train_list.append(rest[index2])
    train = train_list[0]
    for i in range(3):
        train.cat(train_list[i+1])

    dataset_train = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    dataset_validation = DataLoader(val, batch_size=args.batch_size, shuffle=True)

    logging.basicConfig(filename="lstm_ae_train_"+attempt_name+"_idx_"+str(index)+".log", filemode='w', format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

    # start training
    evaluation_loss = []
    train_loss = []
    #start training
    logging.info('start training')
    best_loss = 99999
    best_accuracy = 0.0

    comment = f'gradient_clip = {clip} lr = {learning_rate} hidden_dim = {hidden_dim} ephochs = {epochs} index = {index}'
    tb = SummaryWriter(log_dir=os.path.join(stats_file, comment))
    for epoch in range(epochs):  # loop over the dataset multiple times
        if is_prediction_task:
            train_loss, train_acc = model.train_model(dataset_train, device, optimizer, criterion, classification_criterion, clip, input_size, seq_len)
            val_loss, val_acc = model.evaluate_model(dataset_validation, device, criterion, classification_criterion, input_size, seq_len)
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(model.state_dict(), save_path)
            tb.add_scalar("Train Loss", train_loss, epoch)
            tb.add_scalar("Validation Loss", val_loss, epoch)
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
            logging.info('finished epoch, best loss {}'.format(epoch, best_loss))

        if(epoch%100==0):
            print('train_loss ',train_loss)
            print('val_loss ',val_loss)
            print('epoch ',epoch)

    model.load_state_dict(torch.load(save_path))
    if is_prediction_task:
        test_loss = model.evaluate_model(dataset_test, device, criterion, classification_criterion, input_size, seq_len)
    else:
        test_loss = model.evaluate_model(dataset_test, device, criterion, input_size, seq_len)
    logging.info('test loss {}'.format(test_loss))

print('Finished Training')
