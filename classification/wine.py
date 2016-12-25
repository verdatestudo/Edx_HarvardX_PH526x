'''
Week 3/4 - Case Study 3
from HarvardX: PH526x Using Python for Research on edX

In this case study, we will analyze a dataset consisting of an assortment of wines classified as "high quality" and "low quality"
and will use the k-Nearest Neighbors classifier to determine whether or not other information
about the wine helps us correctly predict whether a new wine will be of high quality.

Last Updated: 2016-Dec-21
First Created: 2016-Dec-20
Python 3.5
Chris
'''

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import random
from collections import Counter

# To ensure that each variable contributes equally to the kNN classifier, we need to standardize the data.
# First, from each variable in numeric_data, subtract its mean.
# Second, for each variable in numeric_data, divide by its standard deviation.
# Store this again as numeric_data.
#
# Principal component analysis is a way to take a linear snapshot of the data from several different angles,
# with each snapshot ordered by how well it aligns with variation in the data.
#
# Use the PCA function in the scikit-learn (sklearn) library to find and store the two most informative principal components of the data
# (a matrix with two columns corresponding to the principal components),
# and store it as pca.
# Use the fit and transform methods on numeric_data
# to extract the first two principal components
# and store them as principal_components.

def mine(numeric_data):
    means = numeric_data.mean(axis = 0)
    stds = numeric_data.std(axis = 0)

    numeric_data = numeric_data.subtract(means)
    numeric_data = numeric_data.divide(stds)

    pca = PCA(n_components=2)

    principal_components = pca.fit_transform(numeric_data)

    return principal_components

def yours(numeric_data):
    '''
    note: add ddof=1 to np.std to get similar results to pandas.
    See: http://stackoverflow.com/questions/24984178/different-std-in-pandas-vs-numpy
    '''
    numeric_data = (numeric_data - np.mean(numeric_data, axis=0)) / np.std(numeric_data, axis=0)

    pca = PCA(2)
    principal_components = pca.fit(numeric_data).transform(numeric_data)

    return numeric_data, principal_components

def plot(principal_components):
    '''
    Plot the first two principal components. The high and low quality wines will be colored using red and blue, respectively.
    How well are the two groups of wines separated by the first two principal components?
    '''
    observation_colormap = ListedColormap(['red', 'blue'])
    x = principal_components[:,0] # Enter your code here!
    y = principal_components[:,1] # Enter your code here!

    plt.title("Principal Components of Wine")
    plt.scatter(x, y, alpha = 0.2,
        c = data['high_quality'], cmap = observation_colormap, edgecolors = 'none')
    plt.xlim(-8, 8); plt.ylim(-8, 8)
    plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
    plt.savefig('wine_1.png')
    plt.show()

def accuracy(predictions, outcomes):
    '''
    We are now ready to fit the wine data to our kNN classifier.
    Create a function accuracy(predictions, outcomes)
    that takes two lists of the same size as arguments and
    returns a single number, which is the percentage of elements that are equal for the two lists.
    '''
    assert predictions.size == outcomes.size
    return sum(predictions == outcomes) / predictions.size * 100

def distance(a, b):
    '''
    Return the Euclidean distance between two points given as numpy arrays.
    '''
    return np.linalg.norm(a - b)

def find_nearest_neighbors(p, points, k=5):
    '''
    Find the k nearest neighbors for point p and return the indices.
    '''
    distances = np.array([distance(x, p) for x in points])
    ind = np.argsort(distances)
    return ind[0:k]

def majority_vote(votes):
    '''
    Takes a sequence votes and returns the mode (random choice if tied).
    '''
    # mode, count = ss.mstats.mode(votes) # alternative but does not choose an element at random if tied.
    vote_count = Counter(votes)
    return random.choice([x[0] for x in vote_count.most_common() if x[1] == vote_count.most_common()[0][1]])

def knn_predict(p, points, outcomes, k=5):
    '''
    Find the k nearest neighbors of point p, and classify it based on outcomes.
    '''
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote(outcomes[ind])



# data = pd.read_csv('https://s3.amazonaws.com/demo-datasets/wine.csv')
data = pd.read_csv('wine.csv')

numeric_data = data.drop('color', axis=1)
numeric_data, principal_components = yours(numeric_data)

# plot(principal_components)

# print(accuracy(np.array([1, 2, 3]), np.array([1, 2, 4])))

# Because most wines in the dataset are classified as low quality,
# one very simple classification rule is to predict that all wines are of low quality.
# Print your result. ~ 80.5%

# print(accuracy(np.zeros_like(data.high_quality), data.high_quality))

# library_predictions = KNeighborsClassifier()
# library_predictions.fit(data, data.high_quality)
# print(library_predictions)

# Use the scikit-learn classifier KNeighborsClassifier to predict which wines are high and low quality and store the result as library_predictions.
# Use accuracy to find the accuracy of library_predictions.
# Print your answer. Is this prediction better than the simple classifier in Exercise 6? 99.9%

# knn = KNeighborsClassifier(n_neighbors = 5)
# knn.fit(numeric_data, data['high_quality'])
# library_predictions = knn.predict(numeric_data)
# print(accuracy(library_predictions, data['high_quality']))

# Unlike the scikit-learn function, our homemade kNN classifier does not take any shortcuts in calculating which neighbors are closest to each observation,
# so it is likely too slow to carry out on the whole dataset.
# To circumvent this, use random.sample and range(n_rows) to sample 10 row indices from the dataset.
# In this case, use seed 123 to select the row indices of your sample.
# Store this selection of indices as selection.

# n_rows = data.shape[0]
# random.seed(123)
# selection = random.sample(range(n_rows), 10)

# Complete my_prediction with a numpy array.
# This array will contain predicted values of the wine's quality (i.e., like "high_quality" column in data).
# You need to use the knn_predict() function with k=5 to get these values from a subset of the data array (e.g., where indices match selection).
# Note that selection is already defined from Exercise 8, and knn_predict is already defined as in the Case 3 videos.
# Using the accuracy function, compare these results to the selected rows from the high_quality variable in data. Store these results as percentage.
# Print your answer.

# predictors = np.array(numeric_data)
# outcomes = np.array(data["high_quality"])
# my_predictions = np.array([knn_predict(predictors[p], predictors, outcomes, k=5) for p in selection])
# percentage = accuracy(my_predictions, np.array([outcomes[x] for x in selection]))
# print(percentage)
