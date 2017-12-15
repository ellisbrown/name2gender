import random
import unicodedata
import csv
import string
import time
from os import getcwd

PROJECT_DIR = getcwd().replace("naive_bayes","")

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
    return "{:0>2}h {:0>2}m {:0>2}s".format(int(hours),int(minutes),int(seconds))

def _gender_features(name):
    features = {}
    features["last_letter"] = name[-1].lower()
    features["first_letter"] = name[0].lower()
    for letter in string.ascii_lowercase:
        features["count(%s)" % letter] = name.lower().count(letter)
        features["has(%s)" % letter] = (letter in name.lower())
    # names ending in -yn are mostly female, names ending in -ch ar mostly male, so add 2 more features
    features["suffix2"] = name[-2:]
    features["suffix3"] = name[-3:]
    features["suffix4"] = name[-4:]
    return features

# DATASETS
NLTK_MBEJDA_FN = "data/nltk_mbejda.csv"
NLTK_MBEJDA_BLK_FN = "data/nltk_mbejda_blk.csv"
NLTK_MBEJDA_BLK_MFRAN_FN = "data/nltk_mbejda_blk_mfran.csv"
SHUFFLED_DATASET_FN = PROJECT_DIR + "data/name_gender_dataset.csv"

DATASET_FN = SHUFFLED_DATASET_FN # this is the dataset file used

# Accessors

TRAIN_SPLIT = 0.75
VAL_SPLIT = 0

# TEST_SPLIT = .25 # ASSUME Test = 1 - (train% + val%)

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


TRAINSET, VALSET, TESTSET = split_dataset()

# Manipulation
ALL_LETTERS = string.ascii_lowercase
ALL_GENDERS = ["male", "female"]
N_LETTERS = len(ALL_LETTERS)
N_GENDERS = len(ALL_GENDERS)