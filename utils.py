import scipy.io
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def load_mat(fname):
    data = scipy.io.loadmat(fname)
    X = data['X']
    y = data['y'].flatten()
    return X,y

def get_vocab_dict():
    words = {}
    inv_words = {}
    f = open('data/vocab.txt','r')
    for line in f:
        if line != '':
            (ind,word) = line.split('\t')
            words[int(ind)] = word.rstrip('\n')
            inv_words[word.rstrip('\n')] = int(ind)
    return words, inv_words

def plot_data(X):
    # plot data
    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='blue', marker='x', label='versicolor')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    #plt.savefig('./images/02_06.png', dpi=300)
    plt.show()

def plot_decision_boundary(X, y, clf):
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),
                         np.arange(y_min, y_max, .01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')

    plt.show()
