import matplotlib.pyplot as plt
import argparse
import pandas
import os
from collections import defaultdict

parser = argparse.ArgumentParser(description='visualize learning loss')
parser.add_argument('--train_folder', help='folder with csv files of train loss')
parser.add_argument('--val_folder', help='folder with csv files of validation loss')


args = parser.parse_args()
train_folder = args.train_folder
val_folder = args.val_folder

train_csvs = {}
val_csvs = {}
train_prediction_csv = {}
val_prediction_csv = {}

for root, dirs, files in os.walk(train_folder):
    for file in files:
        tag_name = file.split('-')[3]
        hyper_params = file.split('-')[1]
        if 'Accuracy' in tag_name:
            train_prediction_csv[hyper_params] = pandas.read_csv(os.path.join(train_folder, file))
        else:
            train_csvs[hyper_params] = pandas.read_csv(os.path.join(train_folder, file))

for root, dirs, files in os.walk(val_folder):
    for file in files:
        tag_name = file.split('-')[3]
        hyper_params = file.split('-')[1]
        if 'Accuracy' in tag_name:
            val_prediction_csv[hyper_params] = pandas.read_csv(os.path.join(val_folder, file))
        else:
            val_csvs[hyper_params] = pandas.read_csv(os.path.join(val_folder, file))

plt.figure('Train Reconstruction Loss')
plt.title('Train Reconstruction Loss')
#plt.ylim(top=0.15)
plt.ylim(top=0.01, bottom=0.00895)
for legend, loss in train_csvs.items():
    plt.plot(loss['Step'], loss['Value'], label=legend)

plt.legend()

plt.figure('Validation Reconstruction Loss')
plt.title('Validation Reconstruction Loss')
plt.ylim(top=0.01, bottom=0.00905)
for legend, loss in val_csvs.items():
    plt.plot(loss['Step'], loss['Value'], label=legend)
plt.legend()

plt.figure('Train Accuracy')
plt.title('Train Accuracy')
plt.ylim(bottom=0.85, top=1.005)
for legend, loss in train_prediction_csv.items():
    plt.plot(loss['Step'], loss['Value'], label=legend)

plt.legend()

plt.figure('Validation Accuracy')
plt.title('Validation Accuracy')
plt.ylim(bottom=0.85)
for legend, loss in val_prediction_csv.items():
    plt.plot(loss['Step'], loss['Value'], label=legend)


plt.legend()
plt.show()