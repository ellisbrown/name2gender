# Name2Gender - Char-RNN
See [demo.ipynb](https://github.com/ellisbrown/name2gender/blob/master/rnn/demo.ipynb) for a code usage demonstration and my [Medium post](https://medium.com/@ellisbrown/name2gender-introduction-626d89378fb0) where I elaborate on the problem.

This approach attempts to learn the various gender-revealing sequences without having to explicitly specify them.

### Tensor Representation
In order to represent each character, we create a one-hot vector of size <N_LETTERS> (a one-hot vector is filled with 0s except for a 1 at the index of the current letter, e.g. "c" = <0 0 1 0 0 ... 0>).

### Model
Following the direction of PyTorch’s name nationality classification example, we create a simple network with 2 linear layers operating on an input and hidden state, and a LogSoftmax layer on the output. We use 128 hidden units.
This is a very simple network definition, and likely could be improved by adding more linear layers or better shaping the network.
