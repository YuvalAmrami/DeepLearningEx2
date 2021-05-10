import torch
import numpy
import pickle
from sklearn.model_selection import train_test_split

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
    

