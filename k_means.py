import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


def generate_random_data(size=20):
    # Set three centers, the model should predict similar results
    center_1 = np.array([1,1])
    center_2 = np.array([5,5])
    center_3 = np.array([8,1])

    # Generate random data and center it to the three centers
    data_1 = np.random.randn(size,2) + center_1
    data_2 = np.random.randn(size,2) + center_2
    data_3 = np.random.randn(size,2) + center_3

    X = np.concatenate((data_1, data_2, data_3), axis = 0)
    return X


# run K-means multiple times to get lower
# error and not get stuck in local optimums.
# If K is 2-10 than multiple runs make more sense than
# when K is large.
def run_kmeans(X, K=2, tolerance=1e-4, max_iter=10, n_inits=10):
    cost_history = []
    best_centroids = []
    best_classes = []
    best_cost = None

    for _ in range(n_inits):
        classes, centroids = clustering(X, K, max_iter)
        cost = compute_cost(X, classes, centroids)
        cost_history.append(cost)

        if best_cost > cost or best_cost == None:
            best_cost = cost
            best_centroids = np.copy(centroids)
            best_classes = np.copy(classes)

    plt.plot(cost_history)
    plt.show()

    plot_clusters(X, K, classes, centroids)


def compute_cost(X, classes, centroids):
    m = X.shape[0]
    error = 0
    for i in range(m):
        error += np.linalg.norm(X[i]-centroids[classes[i]], ord=2)
    return error / m


# return the list of K initial centroids
def init(X, K):
    return X[np.random.choice(X.shape[0], K)]


def clustering(X, K, max_iter=10):
    m = X.shape[0]
    centroids = init(X, K)
    classes = np.zeros(m, dtype=int)

    for iter in range(max_iter):

        for i in range(m):
            distances = np.linalg.norm(X[i]-centroids, axis=1, ord=1)
            classes[i] = np.argmin(distances)

        for k in range(K):
            centroids[k] = np.mean(X[classes==k], axis=0)

    return classes, centroids


def plot_clusters(X, K, classes, centroids):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    num_colors = len(colors)
    for i in range(K):
        points = np.array([X[j] for j in range(len(X)) if classes[j] == i])
        plt.scatter(points[:,0], points[:,1], s=7, c=colors[i%num_colors])
    plt.scatter(centroids[:,0], centroids[:,1], marker='o', s=100, c='#050505')
    plt.show()



X = generate_random_data(size=5)
classes, centroids = clustering(X, 2)
plot_clusters(X, 2, classes, centroids)
#run_kmeans(X, K=2)
