import argparse
import dataset_utils
import numpy as np

def create_dataset(mean, sigma, sample_size, num_samples):
    dataset = []
    for i in range(num_samples):
        dataset.append(np.random.normal(mean, sigma, sample_size))
    return dataset


parser = argparse.ArgumentParser(description='train a lstm over a videos dataset')
parser.add_argument('--num_samples', type=int, help='number of samples for dataset')
parser.add_argument('--sample_size', type=int, help='the size of the samples sequence')
parser.add_argument('--output_file_prefix', help='output file prefix')


args = parser.parse_args()


dataset = create_dataset(0.5, 0.1, args.sample_size, args.num_samples)
train, val, test = dataset_utils.split_dataset(dataset)
print(np.min(train))
print(np.max(train))
print(np.mean(train))

dataset_utils.save_dataset('{}_{}.pkl'.format(args.output_file_prefix, 'train'), dataset)
dataset_utils.save_dataset('{}_{}.pkl'.format(args.output_file_prefix, 'val'), dataset)
dataset_utils.save_dataset('{}_{}.pkl'.format(args.output_file_prefix, 'test'), dataset)