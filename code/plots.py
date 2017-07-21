from __future__ import division
import pickle as pkl
import pdb
from ast import literal_eval
import sklearn
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as Math
from sklearn.manifold import TSNE
import pylab as Plot
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.patches as mpatches

fp = open('./data/instances.jsonl')
lines = fp.readlines()

fp2 = open('./data/truth.jsonl')
lines2 = fp2.readlines()

posts = []
truth_d = {}
truth = []
targets = []
titles = []
ids = []
for line in lines2:
        d = literal_eval(line)
        if d['truthClass'] == 'clickbait':
            truth_d[d['id']] = 1
        else:
            truth_d[d['id']] = 0

for line in lines:
        d = literal_eval(line)
        posts.append(d['postText'][0])
        targets.append(d['targetDescription'])
        truth.append(truth_d[d['id']])
        ids.append(d['id'])
        titles.append(d['targetTitle'])

words = []
for post in posts:
    words.extend(post.split(' '))
for desc in targets:
    words.extend(desc.split(' '))
for title in titles:
    words.extend(title.split(' '))
unique_words = list(set(words))

np_posts = Math.array(posts)
np_targets = Math.array(targets)
np_titles = Math.array(titles)
np_words = Math.array(words)
np_truth = Math.array(truth)
np_unique_words = Math.array(unique_words)

tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=word_tokenize,
                                    stop_words=stopwords.words('english'),
                                    max_df=1.0,
                                    min_df=1,
                                    lowercase=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(np_posts)
desc_tfidf_matrix = tfidf_vectorizer.fit_transform(np_targets)
title_tfidf_matrix = tfidf_vectorizer.fit_transform(np_titles)

def Hbeta(D = Math.array([]), beta = 1.0):
    """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

    # Compute P-row and corresponding perplexity
    P = Math.exp(-D.copy() * beta);
    sumP = sum(P);
    H = Math.log(sumP) + beta * Math.sum(D * P) / sumP;
    P = P / sumP;
    return H, P;

def x2p(X = Math.array([]), tol = 1e-5, perplexity = 20.0):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

    # Initialize some variables
    print "Computing pairwise distances..."
    (n, d) = X.shape;
    sum_X = Math.sum(Math.square(X), 1);
    D = Math.add(Math.add(-2 * Math.dot(X, X.T), sum_X).T, sum_X);
    P = Math.zeros((n, n));
    beta = Math.ones((n, 1));
    logU = Math.log(perplexity);

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print "Computing P-values for point ", i, " of ", n, "..."

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -Math.inf;
        betamax =  Math.inf;
        Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))];
        (H, thisP) = Hbeta(Di, beta[i]);

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU;
        tries = 0;
        while Math.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy();
                if betamax == Math.inf or betamax == -Math.inf:
                    beta[i] = beta[i] * 2;
                else:
                    beta[i] = (beta[i] + betamax) / 2;
            else:
                betamax = beta[i].copy();
                if betamin == Math.inf or betamin == -Math.inf:
                    beta[i] = beta[i] / 2;
                else:
                    beta[i] = (beta[i] + betamin) / 2;

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i]);
            Hdiff = H - logU;
            tries = tries + 1;

        # Set the final row of P
        P[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))] = thisP;

    # Return final P-matrix
    print "Mean value of sigma: ", Math.mean(Math.sqrt(1 / beta));
    return P;

def pca(X = Math.array([]), no_dims = 50):
    """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

    print "Preprocessing the data using PCA..."
    (n, d) = X.shape;
    X = X - Math.tile(Math.mean(X, 0), (n, 1));
    (l, M) = Math.linalg.eig(Math.dot(X.T, X));
    Y = Math.dot(X, M[:,0:no_dims]);
    return Y;

def tsne(X = Math.array([]), no_dims = 2, initial_dims = 50, perplexity = 20.0):
    """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

    # Check inputs
    if isinstance(no_dims, float):
        print "Error: array X should have type float.";
        return -1;
    if round(no_dims) != no_dims:
        print "Error: number of dimensions should be an integer.";
        return -1;

    # Initialize variables
    X = pca(X, initial_dims).real;
    (n, d) = X.shape;
    max_iter = 1000;
    initial_momentum = 0.5;
    final_momentum = 0.8;
    eta = 500;
    min_gain = 0.01;
    Y = Math.random.randn(n, no_dims);
    dY = Math.zeros((n, no_dims));
    iY = Math.zeros((n, no_dims));
    gains = Math.ones((n, no_dims));

    # Compute P-values
    P = x2p(X, 1e-5, perplexity);
    P = P + Math.transpose(P);
    P = P / Math.sum(P);
    P = P * 4;          # early exaggeration
    P = Math.maximum(P, 1e-12);

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = Math.sum(Math.square(Y), 1);
        num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y));
        num[range(n), range(n)] = 0;
        Q = num / Math.sum(num);
        Q = Math.maximum(Q, 1e-12);

        # Compute gradient
        PQ = P - Q;
        for i in range(n):
            dY[i,:] = Math.sum(Math.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
        gains[gains < min_gain] = min_gain;
        iY = momentum * iY - eta * (gains * dY);
        Y = Y + iY;
        Y = Y - Math.tile(Math.mean(Y, 0), (n, 1));

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = Math.sum(P * Math.log(P / Q));
            print "Iteration ", (iter + 1), ": error is ", C

        # Stop lying about P-values
        if iter == 100:
            P = P / 4;

    # Return solution
    return Y;

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(tfidf_matrix)
X_embedded = TSNE(n_components=2, perplexity=20.0, verbose=2).fit_transform(X_reduced)
fig = plt.figure(figsize=(10, 10))
colorb = ['red', 'green']
classes = ['clickbait', 'no-clickbait']
red_patch = mpatches.Patch(color='red', label='clickbait')
green_patch = mpatches.Patch(color='green', label='no-clickbait')
for j in range(2):
    for i in range(len(np_posts)):
        if (truth[i] == j):
            plt.scatter(X_embedded[i, 0], X_embedded[i, 1], 15, colorb[j])
plt.legend(handles=[red_patch, green_patch])
plt.savefig('./data/word_plot.png')

X_desc_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(desc_tfidf_matrix)
X_desc_embedded = TSNE(n_components=2, perplexity=20.0, verbose=2).fit_transform(X_desc_reduced)
fig = plt.figure(figsize=(10, 10))
for j in range(2):
    for i in range(len(np_targets)):
        if (truth[i] == j):
            plt.scatter(X_desc_embedded[i, 0], X_desc_embedded[i, 1], 15, colorb[j])
plt.legend(handles=[red_patch, green_patch])
plt.savefig('./data/desc_plot.png')

X_title_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(title_tfidf_matrix)
X_title_embedded = TSNE(n_components=2, perplexity=20.0, verbose=2).fit_transform(X_title_reduced)
fig = plt.figure(figsize=(10, 10))
for j in range(2):
    for i in range(len(np_titles)):
        if (truth[i] == j):
            plt.scatter(X_title_embedded[i, 0], X_title_embedded[i, 1], 15, colorb[j])
plt.legend(handles=[red_patch, green_patch])
plt.savefig('./data/title_plot.png')
