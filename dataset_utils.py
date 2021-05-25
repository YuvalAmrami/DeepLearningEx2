import torch
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import torchvision.datasets as datasets
from torchvision import transforms


transform = transforms.Compose([transforms.ToTensor()])

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


def split_stocks_dataset(stocks, atmp_name=''):
    # stocks.sort(['symbol', 'date']
    names = stocks['symbol']
    names = names[names.duplicated( )== False]
    data = []
    for name in names:
        stock = stocks.loc[stocks['symbol'] == name]
        stock = stock['close'].values
        if (stock.shape == (1007,)):
            data.append(stock)

    print(type(data))
    data = np.array(data);
    print(type(data))

    train, val, test = split_dataset(data)

    save_dataset('{}_{}.pkl'.format(atmp_name, 'train'), train)
    save_dataset('{}_{}.pkl'.format(atmp_name, 'val'), val)
    save_dataset('{}_{}.pkl'.format(atmp_name, 'test'), test)

    train = torch.from_numpy(train).float()
    val = torch.from_numpy(val).float()
    test = torch.from_numpy(test).float()


    # create all possible sequences of length look_back
    # for index in range(size - look_back):
    #     data.append(data_raw[index: index + look_back])
    return train, val, test

