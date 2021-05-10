import argparse
import dataset_utils
import torch
import torch.optim as optim
import torch.nn as nn
import lstm_model
import logging
from torch.utils.data import DataLoader
import pickle


parser = argparse.ArgumentParser(description='train a lstm auto encoder over synthetic dataset')
parser.add_argument('--ephocs', type=int, help='number of ephocs')
parser.add_argument('--optimizer', help='optimizer for algorithm (Adam,SGD)')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--gd_clip', type=int, help='gradient clipping value')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--hidden_state_size', type=int, help='hidden state size of lstm')
parser.add_argument('--dataset_file_prefix', help='dataset file prefix')
parser.add_argument('--stats_file', help='output file for statistics')


args = parser.parse_args()

ephocs = args.ephocs
optimizer = args.optimizer
learning_rate = args.lr
clip = args.gd_clip
batch_size = args.batch_size
hidden_state_size = args.hidden_state_size
dataset_prefix = args.dataset_file_prefix

logging.basicConfig(filename="lstm_ae_train.log", filemode='w', format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dataset_train = DataLoader(dataset_utils.load_dataset('{}_{}.pkl'.format(dataset_prefix, 'train')), batch_size=args.batch_size, shuffle=True)

dataset_validation = DataLoader(dataset_utils.load_dataset('{}_{}.pkl'.format(dataset_prefix, 'val')), batch_size=args.batch_size, shuffle=True)

dataset_test = DataLoader(dataset_utils.load_dataset('{}_{}.pkl'.format(dataset_prefix, 'test')), batch_size=args.batch_size, shuffle=True)

model = lstm_model.LSTM_AE_Model(device, input_size=1, hidden_dim=hidden_state_size)
model.to(device)

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
for epoch in range(ephocs):  # loop over the dataset multiple times
    train_loss.append(model.train_model(dataset_train, device, optimizer, criterion, 50))
    logging.info('train loss {}, epoch {}'.format(train_loss[-1], epoch))
    evaluation_loss.append(model.evaluate_model(dataset_validation, device, criterion, 50))
    logging.info('evaluation loss {}, epoch {}'.format(evaluation_loss[-1], epoch))
    if evaluation_loss[-1] < best_loss:
        best_loss = evaluation_loss[-1]
        torch.save(model.state_dict(), 'best_lstm_ae.model')
    logging.info('best loss {}, epoch {}'.format(best_loss, epoch))

model.load_state_dict(torch.load('best_lstm_ae.model'))
test_loss = model.evaluate_model(dataset_test, device, criterion, 50)

file_data = {}
file_data['test_data'] = test_loss
file_data['val_data'] = evaluation_loss
file_data['train_data'] = train_loss

with open(args.stats_file, 'wb') as f:
    pickle.dump(file_data, f)

print('Finished Training')

