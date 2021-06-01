from dataset_utils import load_mnist_dataset, load_dataset
from lstm_model_predictor import LSTM_AE_Model_pred
from lstm_model_with_classifcation import LSTM_AE_Classification_Model
from dataset_utils import load_mnist_dataset, load_dataset, strip_names, load_dataset_with_name

from lstm_model import LSTM_AE_Model
import argparse
import torch
import torch.nn as nn


parser = argparse.ArgumentParser(description='visualize lstm auto encoder reconstruction from dataset')
parser.add_argument('--model_path', help='model file path')
parser.add_argument('--prediction', type=bool, default=False, help='is prediction lstm')
parser.add_argument('--dataset_type', help='mnist/synthetic')
parser.add_argument('--hidden_dim', type=int, help='hidden state size')
parser.add_argument('--input_size', default=1, type=int, help='input size')

args = parser.parse_args()
model_path = args.model_path
is_prediction = args.prediction
dataset_type = args.dataset_type
hidden_dim = args.hidden_dim
input_size = args.input_size

print(is_prediction)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if dataset_type == 'mnist':
    seq_len = int(784 / input_size)
    if is_prediction:
        model = LSTM_AE_Classification_Model(device, 10, input_size=input_size, hidden_dim=hidden_dim, seq_len=784)
        _, _, dataset = load_mnist_dataset(True)
        dataset, labels = [x[0] for x in dataset], [x[1] for x in dataset]
    else:
        print(input_size)
        model = LSTM_AE_Model(device, input_size=input_size, hidden_dim=hidden_dim, seq_len=784)
        _, _, dataset = load_mnist_dataset(False)

if dataset_type == 'synthetic':
    seq_len = int(50 / input_size)
    model = LSTM_AE_Model(device, input_size=input_size, hidden_dim=hidden_dim, seq_len=50)
    dataset = load_dataset('{}_{}.pkl'.format('synthetic_dataset', 'test'))



if dataset_type == 'snp500':
    if is_prediction:
        seq_len = int(1006 / input_size)
        model = LSTM_AE_Model_pred(device, input_size=input_size, hidden_dim=hidden_dim, seq_len=seq_len)
        dataset = load_dataset_with_name('snp500_test.pkl')
        names, dataset = strip_names(dataset)
        dataset = torch.from_numpy(dataset).float()
    else:
        seq_len = int(1007 / input_size)
        model = LSTM_AE_Model(device, input_size, hidden_dim=hidden_dim, seq_len=seq_len)
        dataset = load_dataset_with_name('snp500_test.pkl')
        names, dataset = strip_names(dataset)
        dataset = torch.from_numpy(dataset).float()




model.load_state_dict(torch.load(model_path))
reconstruction_criterion = nn.MSELoss()

total_loss = 0
total_prediction_loss = 0
total_accuracy = 0
#reconstruction of time series data
with torch.no_grad():
    for i in range(len(dataset)):
        example = dataset[i]
        if is_prediction:
            if dataset_type == 'snp500':
                example = example[:-1]
                input = example.view(1, seq_len, input_size)
                output, prediction = model(input)
            else:
                input = example.view(1, seq_len, input_size)
                output, classification = model(input)
        else:
            input = example.view(1, seq_len, input_size)
            output = model(input)
        #output = output.view(example.shape)
        total_loss += reconstruction_criterion(input, output)
        if dataset_type == 'mnist':
            if is_prediction:
                predicted_label = torch.argmax(classification)
                total_accuracy += (labels[i] == predicted_label)
        elif dataset_type == 'snp500':
            if is_prediction:
                y_true = dataset[i][1:]
                prediction = torch.squeeze(prediction)
                total_prediction_loss += reconstruction_criterion(prediction, y_true)

print(total_accuracy / len(dataset))
print(total_loss / len(dataset))
print(total_prediction_loss / len(dataset))
