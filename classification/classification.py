# Case Study 3

import numpy as np
from collections import Counter
import random
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn import datasets

def distance(a, b):
    '''
    Return the Euclidean distance between two points given as numpy arrays.
    '''
    return np.linalg.norm(a - b)

def majority_vote(votes):
    '''
    Takes a sequence votes and returns the mode (random choice if tied).
    '''
    # mode, count = ss.mstats.mode(votes) # alternative but does not choose an element at random if tied.
    vote_count = Counter(votes)
    return random.choice([x[0] for x in vote_count.most_common() if x[1] == vote_count.most_common()[0][1]])

def test_majority_vote():
    '''
    Test.
    '''
    d = [1, 1, 1, 2, 2, 2, 3, 4, 3, 5, 4, 4]
    vote_counts = majority_vote(d)
    return vote_counts

def find_nearest_neighbors(p, points, k=5):
    '''
    Find the k nearest neighbors for point p and return the indices.
    '''
    distances = np.array([distance(x, p) for x in points])
    ind = np.argsort(distances)
    return ind[0:k]

def knn_predict(p, points, outcomes, k=5):
    '''
    Find the k nearest neighbors of point p, and classify it based on outcomes.
    '''
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote(outcomes[ind])

def plot(p, points):
    '''
    Plot from vid.
    '''
    plt.plot(points[0:4:, 0], points[0:4:, 1], 'ro')
    plt.plot(points[4::, 0], points[4::, 1], 'go')
    plt.plot(p[0], p[1], 'bo')
    plt.axis([0.5, 3.5, 0.5, 3.5])
    plt.show()

def generate_snyth_data(n=50):
    '''
    Create two sorts of points from bivariate normal distributions.
    '''
    points = np.concatenate((ss.norm(0, 1).rvs((n, 2)), ss.norm(1, 1).rvs((n, 2))), axis=0)
    outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n)))
    return (points, outcomes)

def make_prediction_grid(predictors, outcomes, limits, h, k):
    '''
    Classify each point on the prediction grid.
    '''
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)

    prediction_grid = np.zeros(xx.shape, dtype=int)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            prediction_grid[j, i] = knn_predict(p, predictors, outcomes, k)

    return (xx, yy, prediction_grid)

def plot_prediction_grid(xx, yy, prediction_grid, filename, predictors, outcomes):
    """
    Plot KNN predictions for every point on the grid.
    From plot_prediction_grid.py, put here for completion.
    """
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)

def plot_knn_synth_prediction_grid():
    '''
    From vid.
    '''
    k = 3
    filename='knn_synth_{:2d}.png'.format(k)
    limits = (-3, 4, -3, 4)
    h = 0.1
    predictors, outcomes = generate_snyth_data()
    xx, yy, prediction_grid = make_prediction_grid(predictors, outcomes, limits, h, k)
    plot_prediction_grid(xx, yy, prediction_grid, filename, predictors, outcomes)

def plot_iris_prediction_grid():
    '''
    From vid.
    '''
    iris = datasets.load_iris()

    k = 5
    filename='iris_grid.png'
    limits = (4, 8, 1.5, 4.5)
    h = 0.1

    predictors = iris.data[:, 0:2]
    outcomes = iris.target

    xx, yy, prediction_grid = make_prediction_grid(predictors, outcomes, limits, h, k)
    plot_prediction_grid(xx, yy, prediction_grid, filename, predictors, outcomes)

def simple_iris_plot():
    '''
    From vid.
    '''
    iris = datasets.load_iris()
    predictors = iris.data[:, 0:2]
    outcomes = iris.target

    plt.plot(predictors[outcomes==0][:, 0], predictors[outcomes==0][:, 1], 'ro')
    plt.plot(predictors[outcomes==1][:, 0], predictors[outcomes==1][:, 1], 'go')
    plt.plot(predictors[outcomes==2][:, 0], predictors[outcomes==2][:, 1], 'bo')
    plt.savefig('iris.png')

def sklearn_knn():
    '''
    From vid, sklearn knn predictions.
    '''
    from sklearn.neighbors import KNeighborsClassifier
    iris = datasets.load_iris()
    predictors = iris.data[:, 0:2]
    outcomes = iris.target
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(predictors, outcomes)
    sk_predictions = knn.predict(predictors)

    my_predictions = np.array([knn_predict(p, predictors, outcomes, 5) for p in predictors])

    print('Agreement between custom and sklearn predictions: {}'.format(np.mean(sk_predictions == my_predictions) * 100))
    print('Agreement between custom predictions and actual data: {}'.format(np.mean(outcomes == my_predictions) * 100))
    print('Agreement between sklearn predictions and actual data: {}'.format(np.mean(sk_predictions == outcomes) * 100))


###
# points = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]])
# p = np.array([2.5, 2])
# outcomes = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
# plot(p, points)
# print(knn_predict(p, points, outcomes, k=3))
###

###
# n = 20
# points, outcomes = generate_snyth_data(n)
# plt.plot(points[:n, 0], points[:n, 1], 'ro')
# plt.plot(points[n:, 0], points[n:, 1], 'bo')
# plt.show()
###

#plot_knn_synth_prediction_grid()

# simple_iris_plot()
# plot_iris_prediction_grid()

# sklearn_knn()
