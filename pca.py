import os
import numpy as np

__author__ = 'xp'


##two functions to reshape a list of multi-dimensional data into a data matrix

def asRowMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((0, X[0].size ), dtype=X[0].dtype)
    for row in X:
        mat = np.vstack(( mat, np.asarray(row).reshape(1, -1) ))
    return mat


def asColumnMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty(( X[0].size, 0), dtype=X[0].dtype)
    for col in X:
        mat = np.hstack(( mat, np.asarray(col).reshape(-1, 1) ))
    return mat


##PCA
def pca(X, y, num_components=0):
    [n, d] = X.shape
    if ( num_components <= 0) or ( num_components > n):
        num_components = n
    mu = X.mean(axis=0)
    X = X - mu
    if n > d:
        C = np.dot(X.T, X)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X, X.T)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors)
        for i in range(n):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])
    # or simply perform an economy size decomposition
    # eigenvectors , eigenvalues , variance = np.linalg.svd(X.T, full_matrices=False)
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort(- eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # select only num_components
    eigenvalues = eigenvalues[0: num_components].copy()
    eigenvectors = eigenvectors[:, 0: num_components].copy()
    return [eigenvalues, eigenvectors, mu]


# Y=W(X-u)
def project(W, X, mu=None):
    if mu is None:
        return np.dot(X, W)
    return np.dot(X - mu, W)


#X=YW.T+u
def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(Y, W.T)
    return np.dot(Y, W.T) + mu


#normalize data
def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float(( maxX - minX ))
    # scale to [low...high].
    X = X * ( high - low )
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)


##subplot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def create_font(fontname='Tahoma', fontsize=10):
    return {'fontname': fontname, 'fontsize': fontsize}


def subplot(title, images, rows, cols, sptitle="subplot", sptitles=[], colormap=cm.gray, ticks_visible=True,
            filename=None):
    global i
    fig = plt.figure()
    # main title
    fig.text(.5, .95, title, horizontalalignment='center')
    for i in range(len(images)):
        ax0 = fig.add_subplot(rows, cols, ( i + 1))
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        if len(sptitles) == len(images):
            plt.title("%s #%s" % ( sptitle, str(sptitles[i]) ))
        else:
            plt.title("%s #%d" % ( sptitle, (i + 1) ))
        plt.imshow(np.asarray(images[i]),cmap=colormap)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)


import matplotlib.cm as cm
import scipy.io as sio
import numpy as np

# turn the first (at most) 16 eigenvectors into grayscale
# images (note: eigenvectors are stored by column!)

matfn2 = 'C:\\Users\\xp\\Desktop\\matlab1.mat'
data3 = sio.loadmat(matfn2)
x1 = data3['x14']
X = x1
np.shape(X)
X=X.T
np.shape(X)
matfn = 'C:\\Users\\xp\\Desktop\\Assignment 1 @ BD\\image_analysis.mat'
data = sio.loadmat(matfn)
y = data['label']

[D, W, mu] = pca(X, y)


E = []
for i in range(16):
        e = W[:, i].reshape(X[1].shape)
        E=normalize(e, 0, 255)
        for j in range(len(E)):
            E[j]=round(E[j])
        plt.subplot(4,4, i + 1)
        plt.imshow(E[np.newaxis,:].reshape((64, 64)), cmap=plt.cm.gray)



