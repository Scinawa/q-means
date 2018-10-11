
from time import time
import numpy as np
import sys
np.warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DMeans

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from scipy.stats import mode

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


np.random.seed(42)



def create_plot(x, y1, y2):
    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='iterations (<=)', ylabel='voltage (mV)',
           title='Compare k-means vs q-means w.r.t accuracy and iteration')
    ax.grid()

    plt.show()



def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))



def get_data(X, y, delta):
    """
    porco dio

    :return:
    """
    x_dim_iterations = [5, 10, 20, 25, 30, 35]

    for iteration_number in x_dim_iterations:
        # Compute the clusters

        def kmeneazzati(X, iteration_number):
            kmeans = KMeans(n_clusters=10, random_state=0, max_iter=iteration_number)

            cluster_index = kmeans.fit_predict(X)

            labels = np.zeros_like(cluster_index)
            for digit in range(10):
                mask = (cluster_index == digit)
                labels[mask] = mode(digits.target[mask])[0]

            accuracy = accuracy_score(y, labels)

            return accuracy

        def dmeneazzati(X, y, iteration_number, delta=0.01):
            kmeans = DMeans(n_clusters=10, random_state=0, delta = delta, max_iter=iteration_number)

            cluster_index = kmeans.fit_predict(X)

            labels = np.zeros_like(cluster_index)
            for digit in range(10):
                mask = (cluster_index == digit)
                labels[mask] = mode(y[mask])[0] # this is brilliant!


            accuracy = accuracy_score(y, labels)

            return accuracy

        accuracies_kmeans = [kmeneazzati(X, y, iteration_number) for _ in range(5)]
        accuracies_dmeans = [kmeneazzati(X, y, iteration_number) for _ in range(5)]

        print(accuracies_kmeans)
        print(accuracies_dmeans)

        y1 = np.average(accuracies_kmeans)
        y2 = np.average(accuracies_dmeans)

    return y_1, y_2



def compare():
    digits = load_digits()
    data = scale(digits.data)

    n_samples, n_features = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target

    sample_size = 300

    print("n_digits: %d, \t n_samples %d, \t n_features %d"
          % (n_digits, n_samples, n_features))

    print(82 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10, algorithm="full"),
                  name="k-means++", data=data)

    bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
                  name="random", data=data)

    # in this case the seeding of the centers is deterministic, hence we run the
    # kmeans algorithm only once with n_init=1
    pca = PCA(n_components=n_digits).fit(data)
    bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
                  name="PCA-based",
                  data=data)
    print(82 * '_')

    print(30 * '=', "D-means", 30 * "=")

    delta = float(sys.argv[1])

    bench_k_means(DMeans(init='k-means++', n_clusters=n_digits, n_init=10, delta=delta),
                  name="k-means++", data=data)

    bench_k_means(DMeans(init='random', n_clusters=n_digits, n_init=10, delta=delta),
                  name="random", data=data)

    # in this case the seeding of the centers is deterministic, hence we run the
    # kmeans algorithm only once with n_init=1
    pca = PCA(n_components=n_digits).fit(data)
    bench_k_means(DMeans(init=pca.components_, n_clusters=n_digits, n_init=1, delta=delta),
                  name="PCA-based", data=data)

    print(82 * '_')


if __name__ == "__main__":
    print("we do everything with tol tol=0.0001")
    digits = load_digits()
    data = scale(digits.data)

    n_samples, n_features = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target



    get_data(X, y, delta = 0.01)















































# #############################################################################
# Visualize the results on PCA-reduced data
#
# reduced_data = PCA(n_components=2).fit_transform(data)
# kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
# kmeans.fit(reduced_data)
#
# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
#
# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')
#
# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()
