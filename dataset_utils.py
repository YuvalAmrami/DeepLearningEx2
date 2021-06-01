import sklearn
import torch
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import torchvision.datasets as datasets
from torchvision import transforms


transform = transforms.Compose([transforms.ToTensor()])

def load_dataset_with_name(filename):
    dataset = []
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def split_dataset(dataset):
    X_train, X_test = train_test_split(dataset, test_size=0.2, random_state=1)
    X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    return X_train, X_val, X_test

def save_dataset(filename, dataset):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

def load_dataset(filename):
    dataset = []
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return torch.from_numpy(dataset).float()


def load_mnist_dataset(with_labels):
    train = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    if not with_labels:
        train = [x[0] for x in train]
        train, val = train_test_split(train, test_size=0.25, random_state=1)
        test = [x[0] for x in test]
        return train, val, test
    else:
        train, val = train_test_split(train, test_size=0.25, random_state=1)
        return train, val, test

def strip_names(data):
    data_no_names = []
    names = []
    for (da, name) in data:
        data_no_names.append(da)
        names.append(name)
    data_no_names = np.array(data_no_names)
    names = np.array(names)
    return names, data_no_names


def split_stocks_dataset(stocks):
    names = stocks['symbol']
    names = names[names.duplicated( ) == False]
    data = []
    for name in names:
        stock = stocks.loc[stocks['symbol'] == name]
        stock = stock['close'].values
        if (stock.shape == (1007,)):
            sklearn.preprocessing.minmax_scale(stock, feature_range=(0, 1), axis=0, copy=False)
            data.append((stock, name))
    X_rest, X_test = train_test_split(data, test_size=0.2, random_state=1)
    return X_rest, X_test