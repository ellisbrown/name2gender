import random
import string
import math
import time
import nltk
from nltk.corpus import names
from nltk.classify import apply_features
import numpy as np
import pickle
import argparse

from data_util import PROJECT_DIR, time_since, split_dataset, _gender_features
from classify import load_classifier

parser = argparse.ArgumentParser(description='Name2Gender Naive Bayes Classifier Training')
parser.add_argument('--weights', default="nb/naive_bayes_weights", help='File to save weights to within weights subdir')
parser.add_argument('--dataset', default="name_gender_dataset.csv", help='Dataset csv to test on within data subdir')
parser.add_argument('--test_split', default=0.245, type=float, help='Portion of dataset devoted to testing examples (e.g., 0.245)')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")


args = parser.parse_args()
args.weights = PROJECT_DIR+"weights/"+args.weights
args.dataset = PROJECT_DIR+"data/"+args.dataset

_, _, testset = split_dataset(1 - args.test_split, 0, args.dataset, shuffle=False)

def test(testset=testset, weight_file=args.weights):
    """tests classifier on name->gender
    
    Args:
        train: % of examples to train with (e.g., 0.8)
    """
    start = time.time()
    classifier = load_classifier(weight_file)
    
    print("Testing Naive Bayes Classifer on %d examples (%s)" % (len(testset), time_since(start)))
    testset = apply_features(_gender_features, testset, labeled=True)
    acc = nltk.classify.accuracy(classifier, testset)
    print("Testing accuracy is %.2f%% on %d examples (%s)" % (acc * 100, len(testset), time_since(start)))
    return acc

if __name__ == '__main__':
    test()
