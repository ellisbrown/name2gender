import math
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import argparse

from data_util import *
from model import RNN

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Name2Gender RNN Training')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
#parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--end_epoch', default=100, type=int, help='Final epoch of training')
parser.add_argument('--start_epoch', default=1, type=int, help='Begin counting epochs starting from this value')
#parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, help='initial learning rate')
#parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
#parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
#parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=str2bool, help='Print the loss after each batch')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
args = parser.parse_args()



n_hidden = 128
print_every = 25

learning_rate = args.lr # If you set this too high, it might explode. If too low, it might not learn

batch_size = args.batch_size
num_workers = args.num_workers
start_ep = args.start_epoch # Begin counting iterations starting from this value (should be used with resume)
end_ep = args.end_epoch
save_folder = args.save_folder
log_iters = args.log_iters

rnn = RNN(n_letters, n_hidden, n_genders)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()


def _train(name_tensor, gender_tensor):
    hidden = rnn.init_hidden()
    optimizer.zero_grad()
    
    for letter_tensor in name_tensor:
        letter_tensor.data.unsqueeze_(0)
        output, hidden = rnn(letter_tensor, hidden)

    loss = criterion(output, gender_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data[0]


# Keep track of losses for plotting
all_losses = []

def train(dataset=trainset):
    rnn.train()
    print('Loading Dataset...')
    
    dataset = NameGenderDataset(dataset)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=True, collate_fn=name_gender_collate)
    
    start = time.time()
    print("Beginning training...")
    for epoch in range(start_ep, end_ep + 1):

        ep_loss = 0
        
        # fencepost print
        if epoch == start_ep: print('EPOCH %s/%s' % (start_ep, end_ep))
        
        # iterate over all minibatches
        batch_iterator = iter(data_loader)
        batch = 0
        while(True):
            try:
                batch += 1
                batch_loss = 0
                names_tensor, genders_tensor = next(batch_iterator)
                for name_tensor, gender_tensor in zip(names_tensor,genders_tensor):
                    output, loss = _train(name_tensor, gender_tensor)
                    batch_loss += loss
                if log_iters and batch % print_every == 0: print('\tLoss[ ep%d: %.2f | mb%d: %.2f ]  (%s) ' 
                      % (epoch, ep_loss / batch, batch, batch_loss / len(names_tensor), time_since(start)))
                ep_loss += batch_loss
            except StopIteration:
                break
        print('EPOCH %d %d%% (%s) avg loss: %.4f' % (epoch, epoch / end_ep * 100, time_since(start), ep_loss / batch))

        # Add current loss avg to list of losses
        all_losses.append(ep_loss)
        ep_loss = 0
        torch.save(rnn.state_dict(), save_folder + "/gender_rnn_epoch" + repr(epoch) + '.pth')
    torch.save(rnn.state_dict(), save_folder + '/gender_rnn_classification.pth')
    
if __name__ == '__main__':
    train()