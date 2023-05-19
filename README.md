# DeepLearningEx2
A project made for the CS course Practical Deep Learning.
More information is described in the assignment 2 instructions file: PDL211_HW2.pdf

# train example

lstm_ae_mnist.py --epochs 10000 --optimizer Adam --lr 0.0005 --gd_clip 1 --batch_size 500 --hidden_dim 256 --stats_file_prefix lstm_mnist --model_file_prefix lstm_mnist --input_size 1

python lstm_ae_toy.py --epochs 10000 --optimizer SGD --lr 0.001 --gd_clip 1 --batch_size 1000 --hidden_dim 256 --stats_file lstm_toy --model_save_path lstm_toy --input_size 1
