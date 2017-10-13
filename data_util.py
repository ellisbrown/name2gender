import random
import torch.utils.data as data

# DATASETS
NLTK_MBEJDA_FN = "U:\name2gender\nltk_mbejda.csv"
NLTK_MBEJDA_BLK_FN = "U:\name2gender\nltk_mbejda_blk.csv"
NLTK_MBEJDA_BLK_MFRAN_FN = "U:\name2gender\nltk_mbejda_blk_mfran.csv"
DATASET_FN = NLTK_MBEJDA_BLK_MFRAN_FN # this is the dataset file used

# Accessors

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
    names,genders = load_names(filename)
    namelist = list(zip(names,genders))
    if shuffled:
        random.shuffle(namelist)
    return namelist

