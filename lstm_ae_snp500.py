import argparse

from torch.utils.tensorboard import SummaryWriter

import dataset_utils as DU
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from lstm_model import LSTM_AE_Model
from lstm_model_with_classifcation import LSTM_AE_Classification_Model
import logging
from torch.utils.data import DataLoader
import pickle
from lstm_model_predictor import LSTM_AE_Model_pred
from sklearn.model_selection import KFold
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os



parser = argparse.ArgumentParser(description='train a lstm auto encoder over synthetic dataset')
parser.add_argument('--epochs', type=int, default=7000, help='number of epochs')
parser.add_argument('--optimizer', default='Adam', help='optimizer for algorithm (Adam,SGD)')
parser.add_argument('--lr', type=float, default=10**-6, help='learning rate')
parser.add_argument('--gd_clip', type=int, default=1, help='gradient clipping value')
parser.add_argument('--batch_size', type=int, default=500, help='batch size')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dim')
parser.add_argument('--prediction', type=bool, default=False, help='output file for statistics')
# parser.add_argument('--num_of_days_back', default=40, type=int, help='num of days to calculate back, the number of ts')
parser.add_argument('--stats_file_prefix',  default='stas_snp_Re02B', help='output file for statistics')
parser.add_argument('--model_file_prefix',  default='mod_snp_Re02B', help='output file for model')
parser.add_argument('--input_size', default=1, type=int, help='input size')
parser.add_argument('--attempt_name', default='snp_Re02B', help='name for train, validation and test files of this attempt')

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# seq_len = int(1007 / input_size)

if is_prediction_task:
    seq_len = int(1006 / input_size)
    model = LSTM_AE_Model_pred(device, input_size=input_size, hidden_dim=hidden_dim, seq_len=seq_len)
else:
    seq_len = int(1007 / input_size)
    model = LSTM_AE_Model(device, input_size=input_size, hidden_dim=hidden_dim, seq_len=seq_len)

model.to(device)
criterion = nn.MSELoss()
data_rest, data_test = DU.split_stocks_dataset(stocks, attempt_name, False)
_, rest = DU.strip_names(data_rest)

kf = KFold(n_splits=4)

if optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


_, test = DU.strip_names(data_test)
test = torch.from_numpy(test).float()
dataset_test = DataLoader(test, batch_size=args.batch_size, shuffle=True)


#start training
logging.info('start training')
test_loss = 99999
index = 0
index_of_best = 0
save_path2 = save_path

# cross validation
# if is_prediction_task:
#     _, y_test, test = DU.devid_to_x_y_name(data_test)
#
# else:

for train_name_index, val_name_index in kf.split(rest):
    train_name, val_name = rest[train_name_index], rest[val_name_index]

    logging.basicConfig(filename="lstm_ae_train_" + attempt_name+"_"+str(index) + ".log", filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

    # print(index, train_name.shape, val_name.shape)
    # _, val = DU.strip_names(val_name)
    # _, train = DU.strip_names(train_name)
    val = torch.from_numpy(val_name).float()
    train = torch.from_numpy(train_name).float()
    dataset_validation = DataLoader(val, batch_size=args.batch_size, shuffle=True)
    dataset_train = DataLoader(train, batch_size=args.batch_size, shuffle=True)

    if is_prediction_task:
        model = LSTM_AE_Model_pred(device, input_size=input_size, hidden_dim=hidden_dim, seq_len=seq_len)
        model.to(device)

    else:
        model = LSTM_AE_Model(device, input_size=input_size, hidden_dim=hidden_dim, seq_len=seq_len)
        model.to(device)

    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # start corss val section training
    evaluation_loss = []
    train_loss = []
    curr_best_accuracy = 0.0
    curr_best_loss = 99999
    save_path = 'itr_'+str(index)+'_'+save_path2

    comment = f'gradient_clip = {clip} lr = {learning_rate} hidden_dim = {hidden_dim} ephochs = {epochs} cross_val = {index}'
    tb = SummaryWriter(log_dir=os.path.join(stats_file, comment))

    for epoch in range(epochs):  # loop over the dataset multiple times
        # if is_prediction_task:

        train_loss = model.train_model(dataset_train, device, optimizer, criterion, clip, input_size, seq_len)
        val_loss = model.evaluate_model(dataset_validation, device, criterion, input_size, seq_len)
        if val_loss < curr_best_loss:
            curr_best_loss = val_loss
            torch.save(model.state_dict(), save_path)
        tb.add_scalar("Train Loss", train_loss, epoch)
        tb.add_scalar("Validation Loss", val_loss, epoch)
        logging.info('finished epoch, best accuracy {}'.format(epoch, curr_best_accuracy))

        if (epoch%500==0):
            print("epoch: "+str(epoch))
            print("Train Loss")
            print(train_loss)
            print("val_loss")
            print(val_loss)
            print(index)
    # end of for loop
    model.load_state_dict(torch.load(save_path))
    # if is_prediction_task:
    curr_test_loss = model.evaluate_model(dataset_test, device, criterion, input_size, seq_len)
    # else:
    #     curr_test_loss = model.evaluate_model(dataset_test, device, criterion, input_size, seq_len)
    logging.info('test loss {}'.format(curr_test_loss))

    if curr_test_loss < test_loss:
        test_loss = curr_test_loss
        index_of_best = index
        print('best index is {}', index_of_best)

    index = index + 1

print('Finished Training')

