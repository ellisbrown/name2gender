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

parser = argparse.ArgumentParser(description='Name2Gender Naive Bayes Classifier Training')
parser.add_argument('--weights', default="nb/naive_bayes_weights", help='File to save weights to within weights subdir')
parser.add_argument('--dataset', default="name_gender_dataset.csv", help='Dataset csv to train on within data subdir')
parser.add_argument('--train_split', default=0.75, type=float, help='Portion of dataset to devote to training examples (e.g., 0.75)')
parser.add_argument('--val_split', default=0.05, type=float, help='Portion of dataset to devote to validation examples (e.g., 0.025)')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")

args = parser.parse_args()
args.weights = PROJECT_DIR+"weights/"+args.weights
args.dataset = PROJECT_DIR+"data/"+args.dataset

trainset, valset, _ = split_dataset(args.train_split, args.val_split, args.dataset, shuffle=False)

def train(trainset=trainset, valset=valset, weight_file=args.weights):
    """trains classifier on name->gender
    
    Args:
        trainset: list of name->gender tuple pairs for training
        valset (opt): list of name->gender tuple pairs to validation
        weight_file: filename to save classifer weights

    """

    start = time.time()
    print("Training Naive Bayes Classifer on %d examples (%s)" % (len(trainset), time_since(start)))
    
    trainset = apply_features(_gender_features, trainset, labeled=True)
    classifier = nltk.NaiveBayesClassifier.train(trainset)
    
    print("Training complete. (%s)" % (time_since(start)))
    
    # validation
    if valset is not None and len(valset) > 0: 
        valset = apply_features(_gender_features, valset, labeled=True)
        acc = nltk.classify.accuracy(classifier, valset)
        print("Validation accuracy is %.2f%% on %d examples (%s)" % (acc * 100, len(valset), time_since(start)))

    # save weights
    with open(weight_file, 'wb') as f:
        pickle.dump(classifier, f)
        f.close()
        print('Weights saved to "%s"' % (args.weights))

if __name__ == '__main__':
    train()
    
        