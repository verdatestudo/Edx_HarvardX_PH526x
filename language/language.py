
# Case Study 2

from collections import Counter
import os
import pandas as pd
import matplotlib.pyplot as plt

def count_words(text):
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

def find_quote(quote, book_text):
    '''
    Find location of a quote string in a book.
    '''
    ind = text.find('What\'s in a name?')
    return text[ind:ind + 1000]

def get_book_stats():
    '''
    Put the book stats from the sample books into a pandas table.
    '''
    book_dir = './Books'
    stats = pd.DataFrame(columns=['language', 'author', 'title', 'length', 'unique'])
    title_num = 1

    for language in os.listdir(book_dir):
        for author in os.listdir(book_dir + '/' + language):
            for title in os.listdir(book_dir + '/' + language + '/' + author):
                input_file = book_dir + '/' + language + '/' + author + '/' + title
                print('Reading... {}'.format(input_file))
                text = read_book(input_file)
                num_unique, counts = word_stats(count_words(text))
                stats.loc[title_num] = language, author.capitalize(), title.replace('.txt', ''), sum(counts), num_unique
                title_num += 1
    return stats

def language_plots(stats):
    '''
    Plot data for different languages.
    '''
    plt.figure(figsize = (10, 10))
    subset = stats[stats.language == 'English']
    plt.loglog(subset.length, subset.unique, 'o', label='English', color='crimson')
    subset = stats[stats.language == 'French']
    plt.loglog(subset.length, subset.unique, 'o', label='French', color='forestgreen')
    subset = stats[stats.language == 'German']
    plt.loglog(subset.length, subset.unique, 'o', label='German', color='orange')
    subset = stats[stats.language == 'Portuguese']
    plt.loglog(subset.length, subset.unique, 'o', label='Portuguese', color='blueviolet')
    plt.legend()
    plt.xlabel('Book length')
    plt.ylabel('Number of unique words')
    plt.savefig('lang_plot.png')


stats = get_book_stats()

# print(find_quote('What\'s in a name?', read_book('./Books/English/shakespeare/Romeo and Juliet.txt')))
# print(find_quote('What\'s in a name?', read_book('./Books/German/shakespeare/Romeo und Julia.txt')))

# num_unique, counts = word_stats(count_words(text))
# num_unique2, counts2 = word_stats(count_words(text2))

# language_plots(stats)
