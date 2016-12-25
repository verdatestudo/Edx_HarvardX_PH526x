'''
Week 3/4 - Case Study 2
from HarvardX: PH526x Using Python for Research on edX

In this case study, we will find and plot the distribution of word frequencies for each translation of Hamlet.
Perhaps the distribution of word frequencies of Hamlet depends on the translation --- let's find out!

For these exercises, functions count_words_fast, read_book, and word_stats are already defined as in the Case 2 Videos (Videos 3.2.x).

Last Updated: 2016-Dec-20
First Created: 2016-Dec-20
Python 3.5
Chris
'''

from collections import Counter
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def count_words_fast(text):
    '''
    Count the number of words in a text and store as a Counter object.
    '''
    skips = ['.', ',', ';', ':', '\'', '\"']
    text = ''.join(ch for ch in text.lower() if ch not in skips)
    return Counter(text.split(' '))

def read_book(title_path):
    '''
    Read a book and return it as a string.
    '''
    with open(title_path, 'r', encoding='utf8') as current_file:
        text = current_file.read()
        text = text.replace('\n', '').replace('\r', '')
    return text

def word_stats(word_counts):
    '''
    Return number of unique words and word frequencies.
    '''
    return (len(word_counts), word_counts.values())

def word_count_distribution(text):
    '''
    Text is a string from a book.
    Outputs a dictionary with items corresponding to the count of times a collection of words appears in
    the translation and values corresponding to the number of number of words that appear with that frequency.
    '''
    word_counts = count_words_fast(text)
    count_distribution = Counter(word_counts.values())
    return count_distribution

def more_frequent_me(distribution):
    '''
    Takes a word frequency dictionary and outputs a dictionary with the same keys, and values
    corresponding to the fraction of words that occur with more frequency than that key.
    '''
    keys = distribution.keys()
    mydict = {}
    for key in keys:
        myadd = 0
        for key2 in keys:
            if key2 > key:
                myadd += distribution[key2]
        mydict[key] = myadd / sum(distribution.values())
    return mydict

def more_frequent(distribution):
    '''
    Provided answer.
    Takes a word frequency dictionary and outputs a dictionary with the same keys, and values
    corresponding to the fraction of words that occur with more frequency than that key.
    '''
    counts = sorted(distribution.keys())
    sorted_frequencies = sorted(distribution.values(), reverse = True)
    cumulative_frequencies = np.cumsum(sorted_frequencies)
    more_frequent = 1 - (cumulative_frequencies / cumulative_frequencies[-1])
    return dict(zip(counts, more_frequent))

def test_functions():
    text = 'Romeo and Juliet Romeo Romeo and and and Romeo by by by tea tea tea'
    distribution = word_count_distribution(text)
    print(distribution)
    print(more_frequent(distribution))

def get_hamlets_distribution():
    '''
    For each translation of hamlets, get distribution from previous functions.
    '''
    hamlets =  pd.DataFrame(columns=['language', 'distribution']) ## Enter code here! ###
    book_dir = './language/Books'
    title_num = 1
    for language in os.listdir(book_dir):
        for author in os.listdir(book_dir + '/' + language):
            for title in os.listdir(book_dir + '/' + language + '/' + author):
                if "Hamlet" in title:
                    text = read_book(book_dir + '/' + language + '/' + author + '/' + title)
                    distribution = word_count_distribution(text) ## Enter code here! ###
                    hamlets.loc[title_num] = language, distribution
                    title_num += 1
    return hamlets

def plot():
    '''
    Plot hamlets.
    '''
    hamlets = get_hamlets_distribution()
    colors = ["crimson", "forestgreen", "blueviolet"]
    handles, hamlet_languages = [], []
    for index in range(hamlets.shape[0]):
        language, distribution = hamlets.language[index+1], hamlets.distribution[index+1]
        dist = more_frequent(distribution)
        plot, = plt.loglog(sorted(list(dist.keys())),sorted(list(dist.values()),
            reverse = True), color = colors[index], linewidth = 2)
        handles.append(plot)
        hamlet_languages.append(language)
    plt.title("Word Frequencies in Hamlet Translations")
    xlim    = [0, 2e3]
    xlabel  = "Frequency of Word $W$"
    ylabel  = "Fraction of Words\nWith Greater Frequency than $W$"
    plt.xlim(xlim); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend(handles, hamlet_languages, loc = "upper right", numpoints = 1)
    # show your plot using `plt.show`!
    plt.show()


plot()
