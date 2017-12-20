import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    """Recurrent Neural Network
    original source: https://goo.gl/12wiKB

    Simple implementation of an RNN with two linear layers and a LogSoftmax
    layer on the output

    Args:
        input_size: (int) size of data
        hidden_size: (int) number of hidden units
        output_size: (int) size of output
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        combined = torch.cat((input.float(), hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, cuda=False):
        ret = torch.zeros(1, self.hidden_size)
        if cuda:
            ret.cuda()
        return Variable(ret)
