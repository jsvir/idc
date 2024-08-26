import numpy as np
from munkres import Munkres
import torch
from sklearn.metrics import confusion_matrix, jaccard_score
from scipy.spatial import distance_matrix
from tqdm import tqdm
from sklearn.svm import LinearSVC


def generalizability(X_train, gates_train, Y_train, X_test, gates_test, Y_test):
    """
    How  the interpretation of the prediction generalizes to other simple prediction models, e.g. Linear Support Vector Classification
    """
    classifier = LinearSVC()
    classifier.fit(X_train * gates_train, Y_train)
    return classifier.score(X_test * gates_test, Y_test)


def faithfulness(gates_i, x, inference_fn, y, num_features=784):
    """
    Are the identified features significant for prediction?
    """
    importance_vec = np.sum(gates_i > 0, axis=0)
    importance_ind = np.where(np.sum(gates_i > 0, axis=0) > 0)[0]
    importance_ind_sort = importance_ind[np.argsort(-importance_vec[importance_vec > 0])]
    mask = np.ones(num_features)
    acc_arr_bad = []
    for i in importance_ind_sort:
        mask[i] = 0
        y_hat = inference_fn(x * mask)
        if isinstance(y_hat, torch.Tensor):
            y_hat = y_hat.cpu().numpy()
        mean_val = get_accuracy(y_hat, y, 10)
        acc_arr_bad.append(mean_val)
    return np.corrcoef(importance_vec[importance_ind_sort], np.array(acc_arr_bad))[0, 1]


def stability(x, gates, k=2, subset_size=10000,p=2):
    """
    Are explanations to similar samples consistent?
    inputs:
        x_test is N x D matrix of samples
        gates is N x D matrix predicted by STG for x_test
        k is the number of neighbors
    outputs:
        mean Lipchitz constant of the explanation function
    """
    dist_mat_x = distance_matrix(x, x, p=p)
    nn_dist_mat = np.sort(dist_mat_x, axis=1)[:, 0:k]
    nn_ind_mat = np.argsort(dist_mat_x, axis=1)[:, 0:k]
    lipchitz_constants = []
    for i in tqdm(range(subset_size)):
        lipchitz_constants.append(max(distance_matrix(gates[nn_ind_mat[i], :], gates[nn_ind_mat[i], :])[0][1:] / nn_dist_mat[i][1:]))
    return np.mean(np.array(lipchitz_constants))


def diversity(y, gates, num_clusters=10, num_features=784):
    """
    How different are the selected variables for instances of distinct classes?
    For formula see appendix A.7 in
    Yang, Junchen, Ofir Lindenbaum, and Yuval Kluger. "Locally sparse neural networks for tabular biomedical data." International Conference on Machine Learning. PMLR, 2022.
    """
    per_matrix = np.zeros((num_clusters, num_clusters))
    all_gates = []
    for i in range(num_clusters):
        indices_p = np.where(y == i)[0]
        onez_p = np.zeros(num_features)
        active_gates = np.where(np.median(gates[indices_p, :], axis=0) > 0)[0]
        onez_p[active_gates] = 1
        all_gates = np.append(all_gates, active_gates)
        for j in range(num_clusters):
            indices_n = np.where(y == j)[0]
            active_gates_n = np.where(np.median(gates[indices_n, :], axis=0) > 0)[0]
            onez_n = np.zeros(num_features)
            onez_n[active_gates_n] = 1
            per_matrix[i, j] = jaccard_score(onez_n, onez_p)

    diversity = 100 * (1 - (per_matrix / (num_clusters * (num_clusters - 1))).sum())
    return diversity


def uniqueness(x, gates, k=2, subset_size=10000, p=2):
    """
    uniqueness of the selected features for similar samples (how granular our explanations are?)
    inputs:
        x_test is N x D matrix of samples
        gates is N x D matrix predicted by STG for x_test
        k is the number of neighbors
    """
    dist_mat_x = distance_matrix(x, x, p=p)
    nn_dist_mat = np.sort(dist_mat_x, axis=1)[:, 0:k]
    nn_ind_mat = np.argsort(dist_mat_x, axis=1)[:, 0:k]
    vals = []
    for i in tqdm(range(subset_size)):
        vals.append(min(distance_matrix(gates[nn_ind_mat[i], :], gates[nn_ind_mat[i], :])[0][1:] / nn_dist_mat[i][1:]))
    return np.mean(np.array(vals))



def get_accuracy(cluster_assignments, y_true, n_clusters):
    '''
    Computes the accuracy based on the provided kmeans cluster assignments
    and true labels, using the Munkres algorithm
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    confusion_mat = confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_mat, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return np.mean(y_pred == y_true)


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels