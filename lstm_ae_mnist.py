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


parser = argparse.ArgumentParser(description='train a lstm auto encoder over synthetic dataset')
parser.add_argument('--epochs', type=int, help='number of ephocs')
parser.add_argument('--optimizer', help='optimizer for algorithm (Adam,SGD)')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--gd_clip', type=int, help='gradient clipping value')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--encoder_hidden_dim', type=int, help='encoder hidden dim')
parser.add_argument('--decoder_hidden_dim', type=int, help='decoder hidden dim')
parser.add_argument('--prediction', type=bool, default=False, help='output file for statistics')
parser.add_argument('--stats_file', help='output file for statistics')
parser.add_argument('--model_save_path', help='output file for model')


args = parser.parse_args()

ephocs = args.epochs
optimizer = args.optimizer
learning_rate = args.lr
clip = args.gd_clip
batch_size = args.batch_size
encoder_hidden_dim = args.encoder_hidden_dim
decoder_hidden_dim = args.decoder_hidden_dim
is_prediction_task = args.prediction
save_path = args.model_save_path
print(is_prediction_task)

logging.basicConfig(filename="lstm_ae_train.log", filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if is_prediction_task:
    model = LSTM_AE_Classification_Model(device, 10, input_size=1, encoder_hidden_dim=encoder_hidden_dim, decoder_hidden_dim=decoder_hidden_dim)
    model.to(device)
    classification_criterion = nn.CrossEntropyLoss()
    train, val, test = dataset_utils.load_mnist_dataset(True)
else:
    model = LSTM_AE_Model(device, input_size=1, encoder_hidden_dim=encoder_hidden_dim, decoder_hidden_dim=decoder_hidden_dim)
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
evaluation_loss = []
train_loss = []

logging.info('start training')
best_loss = 99999
best_accuracy = 0.0
for epoch in range(ephocs):  # loop over the dataset multiple times
    if is_prediction_task:
        train_loss.append(model.train_model(dataset_train, device, optimizer, criterion, classification_criterion, clip, 1, 784))
        evaluation_loss.append(model.evaluate_model(dataset_validation, device, criterion, classification_criterion, 1, 784))
        if evaluation_loss[-1][1] > best_accuracy:
            best_accuracy = evaluation_loss[-1][1]
            torch.save(model.state_dict(), save_path)
        logging.info('train loss {}, accuracy : {}, epoch {}'.format(train_loss[-1][0], train_loss[-1][1], epoch))
        logging.info('evaluation loss {}, accuracy : {}, epoch {}'.format(evaluation_loss[-1][0], evaluation_loss[-1][1], epoch))
    else:
        train_loss.append(model.train_model(dataset_train, device, optimizer, criterion, clip, 1, 784))
        evaluation_loss.append(model.evaluate_model(dataset_validation, device, criterion, 1, 784))
        if evaluation_loss[-1] < best_loss:
            best_loss = evaluation_loss[-1]
            torch.save(model.state_dict(), save_path)
        logging.info('train loss {}, epoch {}'.format(train_loss[-1], epoch))
        logging.info('evaluation loss {}, epoch {}'.format(evaluation_loss[-1], epoch))

model.load_state_dict(torch.load(save_path))
if is_prediction_task:
    test_loss = model.evaluate_model(dataset_test, device, criterion, classification_criterion, 1, 784)
else:    
    test_loss = model.evaluate_model(dataset_test, device, criterion, 1, 784)
logging.info('test loss {}'.format(test_loss))

file_data = {}
file_data['test_data'] = test_loss
file_data['val_data'] = evaluation_loss
file_data['train_data'] = train_loss

with open(args.stats_file, 'wb') as f:
    pickle.dump(file_data, f)

print('Finished Training')

