import matplotlib.pyplot as plt
import argparse
import pickle
import pandas
import os

parser = argparse.ArgumentParser(description='visualize learning loss')
parser.add_argument('--train_csv', help='csv file of train loss')
parser.add_argument('--val_csv', help='csv file of test loss')


args = parser.parse_args()
train_csv = args.train_csv
val_csv = args.val_csv

train = pandas.read_csv(train_csv)
val = pandas.read_csv(val_csv)

plt.plot(train['Step'], train['Value'], label='train loss')
plt.plot(val['Step'], val['Value'], label='val loss')

plt.legend()
plt.show()