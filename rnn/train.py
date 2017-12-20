import math
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse

from data_util import *
from model import RNN

parser = argparse.ArgumentParser(description='Name2Gender RNN Training')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
#parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--end_epoch', default=100, type=int, help='Final epoch of training')
parser.add_argument('--start_epoch', default=1, type=int, help='Begin counting epochs starting from this value')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
#parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=str2bool, help='Print the loss after each batch')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--save_name', default='gender_rnn', help='Name of model for saving')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.f:
    torch.multiprocessing.set_start_method("spawn")

args.cuda = not args.disable_cuda and torch.cuda.is_available()
if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

n_hidden = 128

save_nm = PROJECT_DIR + args.save_folder + args.save_name

rnn = RNN(N_LETTERS, n_hidden, N_GENDERS)
optimizer = torch.optim.SGD(rnn.parameters(), lr=args.lr, momentum=args.momentum, 
                            weight_decay=args.weight_decay)
criterion = nn.NLLLoss()

if args.cuda:
    rnn = rnn.cuda()

def _train(name_tensor, gender_tensor):
    hidden = rnn.init_hidden(cuda=True)
    optimizer.zero_grad()

    for letter_tensor in name_tensor:
        letter_tensor.data.unsqueeze_(0)
        output, hidden = rnn(letter_tensor, hidden)

    loss = criterion(output, gender_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data[0]

def train(dataset=TRAINSET, batch_size=args.batch_size, num_workers=args.num_workers,
          start_ep=args.start_epoch, end_ep=args.end_epoch, save_name=save_nm,
          print_every=100, log_iters=args.log_iters):
    rnn.train()
    print('Loading Dataset...')

    dataset = NameGenderDataset(dataset)
    collate_fn = name_gender_collate_cuda if args.cuda else name_gender_collate
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=True, collate_fn=collate_fn, pin_memory=True)

    start = time.time()
    print("Beginning training...")
    for epoch in range(start_ep, end_ep + 1):

        ep_loss = 0

        # fencepost print
        if epoch == start_ep: print('EPOCH %s/%s' % (start_ep, end_ep))

        # iterate over all minibatches
        batch_iterator = iter(data_loader)
        batch = 0
        while True:
            try:
                batch += 1
                batch_loss = 0
                names_tensor, genders_tensor = next(batch_iterator)
                for name_tensor, gender_tensor in zip(names_tensor, genders_tensor):
                    output, loss = _train(name_tensor, gender_tensor)
                    batch_loss += loss
                if log_iters and batch % print_every == 0: print('\tLoss[ ep%d: %.2f | mb%d: %.2f ]  (%s) ' 
                      % (epoch, ep_loss / batch, batch, batch_loss / len(names_tensor), time_since(start)))
                ep_loss += batch_loss
            except StopIteration:
                break
        print('EPOCH %d %d%% (%s) avg loss: %.4f' % (epoch, epoch / end_ep * 100, time_since(start), ep_loss / batch))

        ep_loss = 0
        torch.save(rnn.state_dict(), save_name + '_epoch' + repr(epoch) + '.pth')
    torch.save(rnn.state_dict(), save_name + '_classification.pth')

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    train()
