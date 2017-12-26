# Name2Gender - Char-RNN
See [demo.ipynb](https://github.com/ellisbrown/name2gender/blob/master/rnn/demo.ipynb) for a code usage demonstration and my [Medium post](https://medium.com/@ellisbrown/name2gender-introduction-626d89378fb0) where I elaborate on the problem.

This approach attempts to learn the various gender-revealing sequences without having to explicitly specify them.

### Tensor Representation
In order to represent each character, I create a one-hot vector of size `<N_LETTERS>` (a one-hot vector is filled with 0s except for a 1 at the index of the current letter, e.g. `"c" = <0 0 1 0 0 ... 0>)`.

### Model
The RNN module is based on PyTorch’s [name nationality classification example](https://goo.gl/BB7h2A).

It is a simple network with 2 linear layers operating on an input and hidden state, and a LogSoftmax layer on the output. I use 128 hidden units.

![RNN Module Structure](https://i.imgur.com/Z2xbySO.png)

See [rnn/**model.py**](https://github.com/ellisbrown/name2gender/blob/a650d012bf20d11cf5433cecf51e18e9178695de/rnn/model.py#L5) for the implementation.

## Results
I split the dataset with a 70/30 train-test split (~95k training names, ~40.6k testing names). The best  testing accuracy I was able to achieve was around **75.4% accuracy**. I did not spend much time tweaking hyper-parameters for better results.
