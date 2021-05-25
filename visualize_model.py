from torch.utils.tensorboard import SummaryWriter
from lstm_model import LSTM_AE_Model
import torch
from torch.autograd import Variable

writer = SummaryWriter('runs/visualize_toy_model')

device = torch.device("cpu")

net = LSTM_AE_Model(device, 1)
net.to(device)

random_input = Variable(torch.rand(5, 50, 1))
random_input.to(device)

writer.add_graph(net)
writer.close()