from __future__ import division
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn import svm, preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np
import pdb

#labels - 0 for action, 1 for adventure, 2 for animation, 3 biography, 4 for comedy

features_dir = ['./FC7_Features_clickbait.txt']

def change(k):
    
    n = k.shape[0]
    one_n = np.ones((n,n))/n
    k = k - one_n.dot(k) - k.dot(one_n) + one_n.dot(k).dot(one_n)
    
    return k


def reading():                                                          
                                                                        
    f = open(features_dir[0], 'r')

    print "Reading data"                 
    first = f.readline().split()
    first = list(map(float, first))                                       
                                                                        
    data = np.asarray(first)
    train_count = 11067
    index = 1

    for i in range(len(features_dir)):

        print "Reading file", features_dir[i]
        
        for line in f:
            print "row no", index
            l = line.split()
            l = list(map(float, l))                                           
            data = np.vstack((data, l))
            index += 1
            if index >= train_count:
                break

        f.close()
        if (i+1) < len(features_dir):
            f = open(features_dir[i+1], 'r')
            index = 0

    print "Reading Complete"

    return data

def kernel_pca(data, gamma, n_components):

    print "PCA running"
    
    sq_dists = pdist(data, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    k = exp(-gamma*mat_sq_dists)
    n = k.shape[0]
    one_n = np.ones((n,n))/n
    k = k - one_n.dot(k) - k.dot(one_n) + one_n.dot(k).dot(one_n)
    eigvals, eigvecs = eigh(k)

    x_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
    print "PCA complete"
    return x_pc, k

if __name__ == "__main__":

    gamma = 1e-8

    data = reading()
    data_copy = np.copy(data)
    data3 = np.copy(data)
    
    e, k = kernel_pca(data, gamma, 300)
    
    #svm part being done - testing

    new_data = k.dot(e)
