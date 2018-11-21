
import sys
import itertools
import random
from time import time

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


#import seaborn as sns; sns.set()  # for plot styling

import numpy as np

from scipy.stats import mode

import sklearn as sk
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DMeans

from mnist import MNIST

import importlib
importlib.reload(sk)


np.warnings.filterwarnings('ignore')

mndata = MNIST('/home/scinawa/workspace/hackedkit/python-mnist/data')

X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

X_train, y_train = np.array(X_train), np.array(y_train)
X_test,  y_test  = np.array(X_test),  np.array(y_test)

np.random.seed(42)

delta = 0.1
delta_2 = 0.2
delta_3 = 0.4

n_digits = 10



def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y_train, estimator.labels_),
             metrics.completeness_score(y_train, estimator.labels_),
             metrics.v_measure_score(y_train, estimator.labels_),
             metrics.adjusted_rand_score(y_train, estimator.labels_),
             metrics.adjusted_mutual_info_score(y_train,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_, metric='euclidean',)))



print("PCA preprocessing")
dimred = PCA(n_components=25)
X_train_dr_ = dimred.fit_transform(X_train)
X_test_dr_ = dimred.transform(X_test)






### norms normalization on DR
norms = np.linalg.norm(X_train_dr_, axis=1)
print("The biggest norm is  {}, the smallest is {} ".format(max(norms), min(norms)))
X_train_dr = np.apply_along_axis(np.divide, 1, X_train_dr_, min(norms))
X_test_dr = np.apply_along_axis(np.divide, 1, X_test_dr_, min(norms))
norms = np.linalg.norm(X_train_dr, axis=1)
print("The biggest norm after normalization is {}".format(max(norms)))




### norms normalization on non DR
norms_2 = np.linalg.norm(X_train, axis=1)
print("The biggest norm of NON-DR is  {}, the smallest is {} ".format(max(norms_2), min(norms_2)))
X_train = np.apply_along_axis(np.divide, 1, X_train, min(norms_2))
X_test = np.apply_along_axis(np.divide, 1, X_test, min(norms_2))
norms_2 = np.linalg.norm(X_train, axis=1)
print("The biggest norm of NON-DR after normalization is {}".format(max(norms_2)))


v, eigval, u = np.linalg.svd(X_train, full_matrices=False)
# we start to get very small singular values at 712.
print("The gap ->", eigval[710], eigval[712])
print("Condition number of non-dr-thresholded data is {}".format(eigval[0]/eigval[710]))






print('init \t\tinertia \t homo\t compl\t v-meas\t ARI \t AMI \t silhouette')

#bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10, algorithm="full"),
#                  name="k-means++", data=data)

#bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
#                  name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
#pca = PCA(n_components=n_digits).fit(data)
#bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1), name="k-means (PCA)", data=data)


bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10, algorithm="full"),
                  name="k-means++", data=X_train_dr)

bench_k_means(DMeans(init='k-means++', verbose=0, max_iter=10, n_clusters=n_digits, n_init=10, delta=delta, squared_distances=1), name="q {}".format(delta), data=X_train_dr)

bench_k_means(DMeans(init='k-means++', verbose=0, n_clusters=n_digits, max_iter=10, n_init=10, delta=delta_2, squared_distances=1), name="q {}".format(delta_2), data=X_train_dr)


bench_k_means(DMeans(init='k-means++', verbose=0, n_clusters=n_digits, n_init=10, max_iter=10, delta=delta_3, squared_distances=1), name="q {}".format(delta_3), data=X_train_dr)



#
#
# bench_k_means(DMeans(init='k-means++', verbose=0, n_clusters=n_digits, n_init=10, delta=delta, squared_distances=0), name="q {} s=0".format(delta), data=data)
#
# bench_k_means(DMeans(init='k-means++', verbose=0, n_clusters=n_digits, n_init=10, delta=delta, squared_distances=0), name="q {} s=0".format(delta), data=data)
#
# bench_k_means(DMeans(init='k-means++', verbose=0, n_clusters=n_digits, n_init=10, delta=delta, squared_distances=0), name="q {} s=0".format(delta_2), data=data)
#
# bench_k_means(DMeans(init='k-means++', verbose=0, n_clusters=n_digits, n_init=10, delta=delta, squared_distances=0), name="q {} s=0".format(delta_2), data=data)
#
#
# print(30 * '=', "D-means delta={}".format(delta), 30 * "=")
# vrb = 0
# bench_k_means(DMeans(init='k-means++', verbose=vrb, n_clusters=n_digits, n_init=10, delta=delta, squared_distances=1), name="d-means++", data=data)
# input("agio")
# bench_k_means(DMeans(init='k-means++', verbose=vrb, n_clusters=n_digits, n_init=10, delta=delta, squared_distances=0), name="d-means++", data=data)
# bench_k_means(DMeans(init='random', verbose=vrb, n_clusters=n_digits, n_init=10, delta=delta, squared_distances=1), name="d-means random", data=data)
#
# import sys
# sys.exit("im done")
# pca = PCA(n_components=n_digits).fit(data)
# bench_k_means(DMeans(init=pca.components_, n_clusters=n_digits, n_init=1, delta=delta), name="d-means (PCA)", data=data)
#
# print(30 * '=', "D-means delta={}".format(delta_2), 30 * "=")
#
#
# bench_k_means(DMeans(init='k-means++', n_clusters=n_digits, n_init=10, delta=delta_2), name="d-means++", data=data, squared_distances=0)
#
# bench_k_means(DMeans(init='random', n_clusters=n_digits, n_init=10, delta=delta_2), name="d-means random", data=data, squared_distances=1)
# # in this case the seeding of the centers is deterministic, hence we run the
# # kmeans algorithm only once with n_init=1
# pca = PCA(n_components=n_digits).fit(data)
#
# bench_k_means(DMeans(init=pca.components_, n_clusters=n_digits, n_init=1, delta=delta_2), name="d-means (PCA)", data=data, squared_distances=1)
