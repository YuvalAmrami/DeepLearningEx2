import matplotlib.pyplot as plt
import argparse
import pickle
import pandas
import os

parser = argparse.ArgumentParser(description='visualize learning loss')
parser.add_argument('--train_folder', help='folder with csv files of train loss')
parser.add_argument('--val_folder', help='folder with csv files of validation loss')


args = parser.parse_args()
train_folder = args.train_folder
val_folder = args.val_folder

train_csvs = {}
val_csvs = {}

for root, dirs, files in os.walk(train_folder):
    for file in files:
        train_csvs[file.split('-')[1]] = pandas.read_csv(os.path.join(train_folder, file))

for root, dirs, files in os.walk(val_folder):
    for file in files:
        val_csvs[file.split('-')[1]] = pandas.read_csv(os.path.join(val_folder, file))

plt.figure(0)
for legend, loss in train_csvs.items():
    plt.plot(loss['Step'], loss['Value'], label=legend)

plt.legend()

plt.figure(1)
for legend, loss in val_csvs.items():
    plt.plot(loss['Step'], loss['Value'], label=legend)

plt.legend()
plt.show()