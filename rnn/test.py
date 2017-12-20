import math
import time
import argparse
import torch
import torch.autograd as autograd
import torch.nn as nn

from data_util import *
from model import RNN

parser = argparse.ArgumentParser(description='Name2Gender RNN Training')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
#parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--log_iters', default=False, type=str2bool, help='Print the loss after each batch')
parser.add_argument('--weights', default='weights/old/gender_rnn_epoch47000.pth', help='Weight state dict to load for testing')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.f:
    torch.multiprocessing.set_start_method("spawn")

args.cuda = not args.disable_cuda and torch.cuda.is_available()
if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


weights = args.weights

n_hidden = 128

batch_size = args.batch_size
num_workers = args.num_workers
log_iters = args.log_iters
weights = PROJECT_DIR + args.weights

print('Loading weights...')
rnn = RNN(N_LETTERS, n_hidden, N_GENDERS)
if args.cuda:
    rnn = rnn.cuda()
rnn.load_state_dict(torch.load(weights))
rnn.eval()

def _evaluate(name_tensor):
    hidden = rnn.init_hidden()

    for letter_tensor in name_tensor:
        letter_tensor.data.unsqueeze_(0)
        output, hidden = rnn(letter_tensor, hidden)

    return output

def predict(name, n_predictions=2):
    output = _evaluate(Variable(name_to_tensor(name, args.cuda)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        gender_index = topi[0][i]
        print('(%.2f) %s' % (value, ALL_GENDERS[gender_index]))
        predictions.append([value, ALL_GENDERS[gender_index]])

    return predictions

def validate(dataset=VALSET):
    dataset = NameGenderDataset(dataset)
    collate_fn = name_gender_collate_cuda if args.cuda else name_gender_collate
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=True, collate_fn=collate_fn, pin_memory=True)

    # iterate over all minibatches
    batch_iterator = iter(data_loader)
    cum = 0
    while(True):
        try:
            names_tensor, genders_tensor = next(batch_iterator)
            for name_tensor, gender_tensor in zip(names_tensor, genders_tensor):
                gt = ALL_GENDERS[gender_tensor.data[0]]
                name = tensor_to_name(name_tensor)
                output = _evaluate(name_tensor)
                topv, topi = output.data.topk(k=1, dim=1, largest=True)
                guess = ALL_GENDERS[topi[0][0]]
                cum += 1 if guess == gt else 0
        except StopIteration:
            break
    acc = cum / len(dataset)
    return acc

# Keep track of losses for plotting
ALL_LOSSES = []

def test(dataset=TESTSET, verbose=log_iters):

    print('Loading Dataset...')

    dataset = NameGenderDataset(dataset)
    collate_fn = name_gender_collate_cuda if args.cuda else name_gender_collate
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=True, collate_fn=collate_fn, pin_memory=True)

    
    print("Beginning testing on %s names:\n" % (len(dataset)))
    start = time.time()
    cum = 0
    
     # iterate over all minibatches
    batch_iterator = iter(data_loader)
    batch = 0
    while(True):
        try:
            batch += 1
            batch_acc = 0
            names_tensor, genders_tensor = next(batch_iterator)
            for name_tensor, gender_tensor in zip(names_tensor,genders_tensor):
                gt = ALL_GENDERS[gender_tensor.data[0]]
                name = tensor_to_name(name_tensor)
                output = _evaluate(name_tensor)
                topv, topi = output.data.topk(k=1, dim=1, largest=True)
                guess = ALL_GENDERS[topi[0][0]]
                correct = '!' if guess == gt else 'X (%s)' % gt
                if verbose: print("\t%s -> %s %s " % (name, guess, correct))
                batch_acc += 1 if guess == gt else 0
            print("%.2f%% minibatch acc: %.4f (%s)" % (batch/(len(dataset) / batch_size), batch_acc / len(names_tensor), time_since(start)))
            cum += batch_acc
        except StopIteration:
            break
    acc = cum / len(dataset)
    print()
    print("TOTAL: %d/%d (%.4f%%)" % (cum, len(dataset), acc*100))
    return acc

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    test()
