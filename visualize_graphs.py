import matplotlib.pyplot as plt
import argparse
import pickle
import os

parser = argparse.ArgumentParser(description='visualize learning loss')
parser.add_argument('--files_prefix', help='data files prefix')
parser.add_argument('--data_folder', help='pkl files folder')
parser.add_argument('--lr', help='learning rate for plotting')


args = parser.parse_args()
data_folder = args.data_folder
prefix = args.files_prefix
lr = args.lr

graphs_files = []
for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.startswith(prefix) and file.endswith('.pkl'):
             graphs_files.append(os.path.join(root, file))


for file in graphs_files:
    with open(file, 'rb') as f: 
        data = pickle.load(f)
        train_loss, val_loss, test_loss = data['train_data'], data['val_data'], data['test_data']
        splitted_file = file[:-4].split('_')
        val_loss = list(filter(lambda x: x < 0.012, val_loss))
        if splitted_file[-3] == lr:
            plt.plot(val_loss[200:], label='{}-{}-{}'.format(splitted_file[-3], splitted_file[-2], splitted_file[-1]))
plt.legend()
plt.show()