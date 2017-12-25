# Name2Gender

Using character sequences in first names to predict gender. This is a quick exploration into the interesting problem; see my Medium post where I elaborate on why it is interesting https://medium.com/@ellisbrown/name2gender-introduction-626d89378fb0.

I have implemented a Naïve-Bayes approach and an Char-RNN approach, which are contained in their respective subdirectories.

### Table of Contents
- <a href='https://goo.gl/1dxe5A'>Medium post</a>
- <a href='#naïve-bayes-naive_bayes'>Naïve Bayes</a>
- <a href='#char-rnn-rnn'>Char-Rann</a>
- <a href='#dataset-data'>Dataset</a>
- <a href='#acknowledgement'>Acknowlegement</a>


## Naïve-Bayes [/naive_bayes](https://github.com/ellisbrown/name2gender/tree/master/naive_bayes)
In this approach, I defined features of first names (last two letters, count of vowels, etc.) to use to learn the genders. I explain this in more detail in [my blog post](https://medium.com/@ellisbrown/name2gender-introduction-626d89378fb0#9dfc) and in the [/naive_bayes subdirectory](https://github.com/ellisbrown/name2gender/blob/master/naive_bayes).

## Char-RNN [/rnn](https://github.com/ellisbrown/name2gender/tree/master/rnn)
In this second approach, I feed characters in a name one by one through a character level recurrent neural network built in PyTorch in the hopes of learning the latent space of all character sequences that denote gender without having to define them a priori. I explain this in more detail in [my blog post](https://medium.com/@ellisbrown/name2gender-introduction-626d89378fb0#019f) in the [/rnn subdirectory](https://github.com/ellisbrown/name2gender/blob/master/rnn).

## Dataset [/data](https://github.com/ellisbrown/name2gender/tree/master/data)
I have aggregated multiple smaller datasets representing various cultures into a large dataset (~135k instances) of gender-labeled first names. See [data/**dataset.ipynb**](https://github.com/ellisbrown/name2gender/blob/master/data/dataset.ipynb) for further information on how I pulled it together. Note: I did not spend a ton of time going through and pruning this dataset, so it is probably not amazing or particularly clean (I would greatly appreciate any PR’s if anyone cares or has the time!).


### Acknowledgement
Below are a bunch of links I found useful:
* http://blog.ayoungprogrammer.com/2016/04/determining-gender-of-name-with-80.html/
* http://www.nltk.org/book/ch06.html
* https://medium.com/towards-data-science/deep-learning-gender-from-name-lstm-recurrent-neural-networks-448d64553044
* https://github.com/spro/practical-pytorch/blob/master/char-rnn-classification/char-rnn-classification.ipynb
* http://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
* http://karpathy.github.io/2015/05/21/rnn-effectiveness/
* https://colah.github.io/posts/2015-08-Understanding-LSTMs/
* https://cs231n.github.io/neural-networks-3/#baby
* https://deeplearning4j.org/lstm.html
* https://github.com/spro/practical-pytorch/blob/master/char-rnn-classification/char-rnn-classification.ipynb
* https://github.com/karpathy/char-rnn
* https://machinelearnings.co/text-classification-using-neural-networks-f5cd7b8765c6


