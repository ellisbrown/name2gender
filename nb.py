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

parser = argparse.ArgumentParser(description='Name2Gender Naive Bayes Classifier')
parser.add_argument('--train', default=None, type=str, help='Set true to train the classifier')
parser.add_argument('--test', default=None, type=str, help='Set true to test the classifier')
parser.add_argument('--name', defualt=
parser.add_argument('--weights', default="weights/nb/naive_bayes_weights", help='Pickled weight state dict to load for testing')
args = parser.parse_args()

NB_WEIGHTS = args.weights

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

def train(trainset=trainset, valset=valset, weight_file=NB_WEIGHTS):
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

    # save weights
    with open(weight_file, 'wb') as f:
        pickle.dump(classifier, f)
        f.close()
    
    print("Training complete. (%s)" % (time_since(start)))
    
    # validation
    if valset is not None and len(valset) > 0: 
        valset = apply_features(_gender_features, valset, labeled=True)
        acc = nltk.classify.accuracy(classifier, valset)
        print("Validation accuracy is %.2f%% on %d examples (%s)" % (acc, len(valset), time_since(start)))

def load_classifier(weight_file=NB_WEIGHTS):
    with open(weight_file, 'rb') as f:
        classifier = pickle.load(f)
        f.close()
    print('Loaded weights from "%s"' % (weight_file))
    return classifier

def classify_name(name, weight_file=NB_WEIGHTS):
    name_ = _gender_features(clean_str(name))
    classifier = load_classifier(weight_file)
    guess = classifier.classify(name_)
    print("%s -> %s" % (name, guess))
    return guess
                   
def prob_classify_name(name, weight_file=NB_WEIGHTS):
    name_ = _gender_features(clean_str(name))
    classifier = load_classifier(weight_file)
    dist = classifier.prob_classify(name_)
    m, f = dist.prob("male"), dist.prob("female")
    d = {m: "male", f: "female"}
    first, last = max(m,f), min(m,f)
    print("%s:\n  (%.2f%%) %s\n  (%.2f%%) %s" % (name, first, d[first], last, d[last]))

def test(testset=testset, weight_file=NB_WEIGHTS):
    """trains classifier on name->gender
    
    Args:
        train: % of examples to train with (e.g., 0.8)
    """
    start = time.time()
    classifier = load_classifier(weight_file)
    
    print("Testing Naive Bayes Classifer on %d examples (%s)" % (len(testset), time_since(start)))
    testset = apply_features(_gender_features, testset, labeled=True)
    acc = nltk.classify.accuracy(classifier, testset)
    print("Testing accuracy is %.2f%% on %d examples (%s)" % (acc, len(testset), time_since(start)))
    return acc

if __name__ == '__main__':
    if args.train is not None:
        train(args.weights)
    
