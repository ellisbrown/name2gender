import random
import string
import math
import time
import nltk
from nltk.corpus import names
from nltk.classify import apply_features
import numpy as np
import pickle
import csv
import argparse

start = time.time()
from data_util import PROJECT_DIR, str2bool, clean_str, time_since, split_dataset, _gender_features

parser = argparse.ArgumentParser(description='Name2Gender Naive Bayes Classification')
parser.add_argument('names', metavar='N', type=str, nargs='*', help="Any number of names to be classified")
parser.add_argument('--weights', default="nb/naive_bayes_weights", help='File to save weights to within weights subdir')
parser.add_argument('--infile', default=None, type=str, help='Plaintext file to read names from')
parser.add_argument('--outfile', default=None, type=str, help='CSV file to store name classifications in')
parser.add_argument('--verbose', default=True, type=str2bool, help="Set to False to prevent printing each classification")
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")


args = parser.parse_args()

args.weights = PROJECT_DIR+"weights/"+args.weights

def load_classifier(weight_file=args.weights, verbose=False):
    with open(weight_file, 'rb') as f:
        classifier = pickle.load(f)
        f.close()
    if verbose: print('Loaded weights from "%s"...\n' % (weight_file))
    return classifier

def _classify(name, classifier, verbose=args.verbose):
    _name = _gender_features(clean_str(name))
    dist = classifier.prob_classify(_name)
    m, f = dist.prob("male"), dist.prob("female")
    d = {m: "male", f: "female"}
    prob = max(m,f)
    guess = d[prob]
    if verbose: print("%s -> %s (%.2f%%)" % (name, guess, prob * 100))
    return guess, prob

def classify(names, weight_file=args.weights):
    classifier = load_classifier(weight_file)
    for name in names:
         _classify(name, classifier)
    print("\nClassified %d names (%s)" % (len(names), time_since(start)))


def write_classifications(names, weight_file=args.weights, outfile=args.outfile):
    classifier = load_classifier(weight_file)
    headers = ["name", "gender", "probability"]
    with open(outfile, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(headers)
        for name in names:
            guess, prob = _classify(name, classifier)
            writer.writerow([name, guess, prob])
        f.close()
    print('\nWrote %d names to "%s" (%s)' % (len(names), outfile, time_since(start)))

def read_names(filename=args.infile):
    with open(filename, 'r') as f:
        names = [clean_str(line.rstrip('\n')) for line in f]
    print("Loaded %d names from %s" % (len(names), filename))
    return names
        

if __name__ == '__main__':
    names = read_names() if args.infile is not None else args.names
    if args.outfile is not None:
        # print to file
        write_classifications(names)
    else:
        # print to screen only
        classify(names)
