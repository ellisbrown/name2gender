import random
import unicodedata
import csv
import string
import time
from os import getcwd
import torch
from torch.utils import data
from torch.autograd import Variable

PROJECT_DIR = getcwd().replace("rnn", "")

# DATASETS
NLTK_MBEJDA_FN = "data/nltk_mbejda.csv"
NLTK_MBEJDA_BLK_FN = "data/nltk_mbejda_blk.csv"
NLTK_MBEJDA_BLK_MFRAN_FN = "data/nltk_mbejda_blk_mfran.csv"
SHUFFLED_DATASET_FN = PROJECT_DIR + "data/name_gender_dataset.csv"

DATASET_FN = SHUFFLED_DATASET_FN # this is the dataset file used

TRAIN_SPLIT = 0.75
VAL_SPLIT = 0

# TEST_SPLIT = .25 # ASSUME Test = 1 - (train% + val%)

# helpers
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')

def clean_str(s):
    uncoded = ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in string.ascii_letters
    )
    return uncoded.lower()

def time_since(since):
    now = time.time()
    s = now - since
    hours, rem = divmod(now-since, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}h {:0>2}m {:0>2}s".format(int(hours), int(minutes), int(seconds))

# data accessors
def load_names(filename=DATASET_FN):
    """loads all names and genders from the dataset

    Args:
        filename (optional): path to the desired dataset
            (default: DATASET_FN)

    Return:
        (names, genders):
            names: list of names - e.g., ["john", "bob", ...]
            genders: list of genders - e.g., ["male", "male", "female", ...]
    """

    names = []
    genders = []

    with open(filename) as csv_data_file:
        csv_reader = csv.reader(csv_data_file)
        for row in csv_reader:
            names.append(row[0])
            genders.append(row[1])

    return names, genders


def load_dataset(filename=DATASET_FN, shuffled=True):
    """Returns the name->gender dataset ready for processing

    Args:
        filename (string, optional): path to dataset file
            (default: DATASET_FN)
        shuffled (Boolean, optional): set to False to return the dataset unshuffled
    Return:
        namelist (list(String,String)): list of (name, gender) records
    """
    names, genders = load_names(filename)
    namelist = list(zip(names, genders))
    if shuffled:
        random.shuffle(namelist)
    return namelist

def split_dataset(train_pct=TRAIN_SPLIT, val_pct=VAL_SPLIT, filename=DATASET_FN, shuffle=False):
    dataset = load_dataset(filename, shuffle)
    n = len(dataset)
    tr = int(n * train_pct)
    va = int(tr + n * val_pct)
    return dataset[:tr], dataset[tr:va], dataset[va:]  # Trainset, Valset, Testset

def dataset_dicts(dataset=load_dataset()):
    name_gender = {}
    gender_name = {}
    for name, gender in dataset:
        name_gender[name] = gender
        gender_name.setdefault(gender, []).append(name)
    return name_gender, gender_name

# data manipulators
def name_to_tensor(name, cuda=False):
    """converts a name to a vectorized numerical input for use with a nn
    each character is converted to a one hot (n, 1, 26) tensor

    Args:
        name (string): full name (e.g., "Ellis Brown")

    Return:
        tensor (torch.tensor)
    """

    name = clean_str(name)
    tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    torch.zeros(len(name), N_LETTERS, out=tensor)
    for li, letter in enumerate(name):
        letter_index = ALL_LETTERS.find(letter)
        tensor[li][letter_index] = 1
    return tensor

def tensor_to_name(name_tensor):
    ret = ""
    for letter_tensor in name_tensor.split(1):
        nz = letter_tensor.data.nonzero()
        if torch.numel(nz) != 0:
            ret += (string.ascii_lowercase[nz[0, 1]])
    return ret

def gender_from_output(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    gender_i = top_i[0][0]
    return ALL_GENDERS[gender_i], gender_i

# def random_training_pair():
#     gender = random.choice(all_genders)
#     name = random.choice(gender_name[gender])
#     gender_tensor = Variable(torch.LongTensor([all_genders.index(gender)]))
#     name_tensor = Variable(name_to_tensor(name))
#     return gender, name, gender_tensor, name_tensor


class NameGenderDataset(data.Dataset):
    def __init__(self, data):
        """data should be a list of (name, gender) string pairs"""
        self.data = data
        self.names, self.genders = zip(*data)

    def __getitem__(self, index):
        return self.names[index], self.genders[index]

    def index_of(self, name):
        return self.names.index(name)

    def __len__(self):
        return len(self.data)

def name_gender_collate(batch):
    """takes a minibatch of names, sorts them in descending order of name length,
    converts each name to a one-hot LongTensor
        -> ( example #, character # in name, character # in alphabet )

    Args:
        batch (list of String tuples): each list item is a labelled example (e.g, ("john","male"))
            e.g, [("john", "male), ("jane", "female"), ... ]

    Return:
        a tuple containing:
            (LongTensor) a batch of names stacked on the 0 dim
                size: (batch size, max name length, length of alphabet)
            (list of Variables containing LongTensors):
                gender annotations for the corresponding name
    """

    # sort batch in descending order of name length, maintaining order of gender list
    batch.sort(key=lambda tup: (len(tup[0]), tup), reverse=True)
    #     print(batch)
    names, genders = zip(*batch)
    # ( name in batch, charcter in name, character in alphabet )
    nms = torch.zeros(len(names), len(names[0]), len(ALL_LETTERS))
    gts = []
    for idx, (name, gender) in enumerate(batch):
        for li, letter in enumerate(clean_str(name)):
            letter_index = ALL_LETTERS.find(letter)
            nms[idx][li][letter_index] = 1
        app = torch.LongTensor([ALL_GENDERS.index(gender)])
        gts.append(Variable(app))
    return Variable(nms), gts

def name_gender_collate_cuda(batch):
    """takes a minibatch of names, sorts them in descending order of name length,
    converts each name to a one-hot LongTensor
        -> ( example #, character # in name, character # in alphabet )

    Args:
        batch (list of String tuples): each list item is a labelled example (e.g, ("john","male"))
            e.g, [("john", "male), ("jane", "female"), ... ]

    Return:
        a tuple containing:
            (LongTensor) a batch of names stacked on the 0 dim
                size: (batch size, max name length, length of alphabet)
            (list of Variables containing LongTensors):
                gender annotations for the corresponding name
    """

    # sort batch in descending order of name length, maintaining order of gender list
    batch.sort(key=lambda tup: (len(tup[0]), tup), reverse=True)
    #     print(batch)
    names, genders = zip(*batch)
    # ( name in batch, charcter in name, character in alphabet )
    nms = torch.cuda.LongTensor()
    torch.zeros(len(names), len(names[0]), len(ALL_LETTERS), out=nms)
    gts = []
    for idx, (name, gender) in enumerate(batch):
        for li, letter in enumerate(clean_str(name)):
            letter_index = ALL_LETTERS.find(letter)
            nms[idx][li][letter_index] = 1
        app = torch.cuda.LongTensor([ALL_GENDERS.index(gender)])
        gts.append(Variable(app))
    return Variable(nms), gts

# constants
TRAINSET, VALSET, TESTSET = split_dataset()

ALL_LETTERS = string.ascii_lowercase
ALL_GENDERS = ["male", "female"]
N_LETTERS = len(ALL_LETTERS)
N_GENDERS = len(ALL_GENDERS)