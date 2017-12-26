# Name2Gender - Naïve-Bayes
See [demo.ipynb](https://github.com/ellisbrown/name2gender/blob/master/naive_bayes/demo.ipynb) for a code usage demonstration and my [Medium post](https://medium.com/@ellisbrown/name2gender-introduction-626d89378fb0) where I elaborate on the problem.

In this approach, I defined features of first names (last two letters, count of vowels, etc.) to use to learn the genders. I explain this in more detail in [my blog post](https://medium.com/@ellisbrown/name2gender-introduction-626d89378fb0#9dfc) and in [my demo notebook](https://github.com/ellisbrown/name2gender/blob/master/naive_bayes/demo.ipynb).

## Features
See the [_gender_features](https://github.com/ellisbrown/name2gender/blob/master/naive_bayes/data_util.py#L42) function for the feature implementation.
#### last letter
* "last_letter": 's'
#### first letter
* "first_letter": 'e'
#### count of letters
* "count(a)": 0, ...
* "count(e)": 1, ...
#### has letter
* "has(a)": False, ...
* "has(e)": True, ...
#### suffixes (last 2, 3, 4 letters of name)
* "suffix2": "is"
* "suffix3": "lis"
* "suffix4": "llis"
