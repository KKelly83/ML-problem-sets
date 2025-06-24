# PCA with SVMs
# Performs PCA on data set first, then uses a dual SVM w/ gaussian kernel to build classifier

import numpy as np
import pandas as pd
import cvxopt as cvxopt
from joblib import Parallel, delayed

# helper to computer gaussian kernel between two matrices
def gaussian_kernel(X1, X2, sigma):
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-sq_dists / (2 * sigma**2))

# train w/ dual SVM w/ a gaussian kernel
def train(X, y, c, sigma_sq):
    total_samples = X.shape[0]
    g_k = gaussian_kernel(X, X, sigma_sq)
    gaussian_kernel_matrix = g_k * (y @ y.T)

    P = cvxopt.matrix(gaussian_kernel_matrix)
    q = cvxopt.matrix(-1 * np.ones(total_samples))
    G = cvxopt.matrix(np.concatenate((-1 * np.eye(total_samples), np.eye(total_samples)), axis=0))
    h = cvxopt.matrix(np.concatenate((np.zeros((total_samples, 1)), c * np.ones((total_samples, 1))), axis=0))
    A = cvxopt.matrix(y.T, tc='d')
    b = cvxopt.matrix(np.zeros(1))
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    return np.array(solution['x']).flatten()


# predictions with respect to SVM
def predict(X_train, y_train, X_test, lagrange_mults, sigma_sq):
    kernel_vals = gaussian_kernel(X_test, X_train, sigma_sq)
    decisions = kernel_vals @ (lagrange_mults * y_train.flatten())
    return np.sign(decisions)


# helper to compare predictions to actual labels
def compute_accuracy(predictions, y):
    return np.mean(predictions == y.flatten())

# zero mean and variance of 1
def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1

    X_norm = (X - mean) / std
    return X_norm

# to perform PCA on training set
def PCA(X):
    X_new = X - np.mean(X, axis=0)
    cov = np.cov(X_new, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvalues, eigenvectors

# get smallest k vals given thresholds
def compute_k(eigenvalues, thresholds):
    total_variance = np.sum(eigenvalues)
    K = {}
    for threshold in thresholds:
        cumulative_variance = 0
        k_z = 0
        for i, eigenvalue in enumerate(eigenvalues):
            cumulative_variance += eigenvalue
            if cumulative_variance / total_variance >= threshold:
                k_z = i + 1
                break
        K[threshold] = k_z
    return K


# splitting up for multiprogramming usage
def PCA_SVM(X_train_proj, train_y, c, sigma, X_val_proj, validation_y, k):
    lagranges = train(X_train_proj, train_y, c, sigma)
    predictions = predict(X_train_proj, train_y, X_val_proj, lagranges, sigma)
    accuracy = compute_accuracy(predictions, validation_y)
    print('Accuracy of the learned classifier on the validation set for k = %5.5f and c = %5.5f and σ2=%5.5f is %5.5f' % (k,
    c, sigma, accuracy))
    return accuracy, c, sigma, k

def no_PCA_SVM(X_train, train_y, c, sigma, X_val, validation_y):
    lagranges = train(X_train, train_y, c, sigma)
    predictions = predict(X_train, train_y, X_val, lagranges, sigma)
    accuracy = compute_accuracy(predictions, validation_y)
    print('Accuracy of the learned classifier on the validation set for c = %5.5f and σ2=%5.5f is %5.5f' % (
    c, sigma, accuracy))
    return accuracy, c, sigma

if __name__ == '__main__':
    # Load data
    train_data = pd.read_csv("gisette_train.data", header=None).values
    validation_data = pd.read_csv("gisette_valid.data", header=None).values
    test_data = pd.read_csv("gisette_test.data", header=None).values

    train_y = train_data[:, 0].reshape(-1, 1)
    train_y[train_y==0] = -1
    train_X = train_data[:, 1:]
    validation_y = validation_data[:, 0].reshape(-1, 1)
    validation_y[validation_y==0] = -1
    validation_X = validation_data[:, 1:]
    test_y = test_data[:, 0].reshape(-1, 1)
    test_y[test_y==0] = -1
    test_X = test_data[:, 1:]

    # normalize all data
    X_train_norm = normalize(train_X)
    X_val_norm = normalize(validation_X)
    X_test_norm = normalize(test_X)

    C_values = [0.001, 0.1, 1, 10, 100, 1000]
    sigma_values = [0.001, 0.1, 1, 10, 100]
    K = {}
    thresholds = [0.99, 0.95, 0.90, 0.80, 0.75]
    ideal_c, ideal_sigma_sq, ideal_k, minimum_accuracy = -1, -1, -1, -999999

    # calculating eigenvalues/vectors

    eigenvalues, eigenvectors = PCA(X_train_norm)

    # determining k values for variance thresholds
    K = compute_k(eigenvalues, thresholds)

    # grab top 6 eigenvalues
    top_six_eigenvalues = eigenvalues[:6]
    print("Top six eigenvalues: ", top_six_eigenvalues)

    print("K values: ", K)

    for k, K_z in K.items():
        V_k = eigenvectors[:, :K_z]

        X_train_proj = X_train_norm @ V_k
        X_val_proj = X_val_norm @ V_k
        X_test_proj = X_test_norm @ V_k


        results = Parallel(n_jobs=-1)(delayed(PCA_SVM)(X_train_proj.copy(), train_y.copy(), c, sigma, X_val_proj.copy(), validation_y.copy(), K_z
                                              ) for c in C_values for sigma in sigma_values)
        for accuracy, c, sigma, k in results:
            if accuracy > minimum_accuracy:
                    ideal_c = c
                    ideal_sigma_sq = sigma
                    ideal_accuracy = accuracy
                    minimum_accuracy = ideal_accuracy
                    ideal_k = k


    print()
    print("Normal SVM:")
    s_ideal_c, s_ideal_sigma_sq, s_minimum_accuracy = -1, -1, -999999
    s_results = []
    for c in C_values:
        for sigma in sigma_values:
            s_results.append(no_PCA_SVM(X_train_norm.copy(), train_y.copy(), c, sigma, X_val_norm.copy(), validation_y.copy()))

    for accuracy, c, sigma in s_results:
        if accuracy > s_minimum_accuracy:
            s_ideal_c = c
            s_ideal_sigma_sq = sigma
            s_ideal_accuracy = accuracy
            s_minimum_accuracy = s_ideal_accuracy


    print("PCA SVM RESULTS")
    print('Ideal c = %5.5f and Ideal sigma = %5.5f' % (ideal_c, ideal_sigma_sq))
    print('Ideal k = ', ideal_k)
    print('accuracy on the test set for the selected classifier %5.3f' % ideal_accuracy)

    print()
    print("NO PCA SVM RESULTS:")
    print('Ideal c = %5.5f and Ideal sigma = %5.5f' % (s_ideal_c, s_ideal_sigma_sq))
    print('accuracy on the test set for the selected classifier %5.3f' % s_ideal_accuracy)


# Alternative to above, PCA with feature selection with rbf kernel

#####################################################################
###########    USED TO RUN PCA WITH FEATURE SELECTION     ###########
#####################################################################

import numpy as np
import sys
import pandas as pd
from cvxopt import matrix, solvers
from joblib import Parallel, delayed
np.set_printoptions(threshold=sys.maxsize)

def load_and_process_data():
    # Load data
    train_data = pd.read_csv("gisette_train.data", header=None).values
    validation_data = pd.read_csv("gisette_valid.data", header=None).values
    test_data = pd.read_csv("gisette_test.data", header=None).values

    train_y = train_data[:, 0].reshape(-1, 1)
    train_y[train_y==0] = -1
    train_X = train_data[:, 1:]
    validation_y = validation_data[:, 0].reshape(-1, 1)
    validation_y[validation_y==0] = -1
    validation_X = validation_data[:, 1:]
    test_y = test_data[:, 0].reshape(-1, 1)
    test_y[test_y==0] = -1
    test_X = test_data[:, 1:]

    # normalize all data
    X_train_norm = normalize(train_X)
    X_val_norm = normalize(validation_X)
    X_test_norm = normalize(test_X)

    return X_train_norm, train_y, X_val_norm, validation_y, X_test_norm, test_y

# zero mean and variance of 1
def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1

    X_norm = (X - mean) / std
    return X_norm

# computing pi
def compute_feature_probabilities(top_k_eigenvectors):
    squared_eigenvectors = top_k_eigenvectors**2
    pi_j = np.mean(squared_eigenvectors, axis=1)
    return pi_j

# sampling s features
def sample_features(pi_j, s):
    # normalizing pi_j
    pi_j = pi_j / np.sum(pi_j)
    sampled_indices = np.random.choice(len(pi_j), size=s, replace=True,p=pi_j).astype(int)
    unique_sampled_indices = np.unique(sampled_indices).astype(int)
    return unique_sampled_indices

# reduce dataset to sampled features
def reduce_dataset(X, sampled_features):
    X_reduced = X[:, sampled_features]
    return X_reduced

# to perform PCA on training set
def PCA(X):
    X_new = X - np.mean(X, axis=0)
    cov = np.cov(X_new, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvalues, eigenvectors

# rbf kernel for use with SVM
def rbf_kernel(X1, X2, sigma):
    sq_dist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-sq_dist / (2 * sigma ** 2))

# train svm wit slack with rbf kernel matrix
def train_svm(X_train, y_train, C, sigma):
    n_samples = X_train.shape[0]

    # compute kernel matrix
    K = rbf_kernel(X_train, X_train, sigma)
    P = matrix(np.outer(y_train, y_train) * K)
    q = matrix(-np.ones((n_samples, 1)))

    # constraints Gx <= h
    G_std = np.diag(-np.ones(n_samples))  # G for alpha >= 0
    G_slack = np.diag(np.ones(n_samples))  # G for alpha <= C
    G = matrix(np.vstack((G_std, G_slack)))
    h = matrix(np.hstack((np.zeros(n_samples), C * np.ones(n_samples))))

    # equality constraint Ax = b
    A = matrix(y_train.T, (1, n_samples), 'd')
    b = matrix(0.0)

    # solve QP problem
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)

    # extract lagrange multipliers
    alpha = np.array(solution['x']).flatten()

    # compute bias term w/ support vectors
    support_indices = (alpha > 1e-5) & (alpha < C - 1e-5)
    y_train = y_train.flatten()
    b = np.mean(y_train[support_indices] - np.dot(K[support_indices], (alpha * y_train).reshape(-1, 1)).flatten())
    return alpha, b

# SVM classifier helper
def train_svm_classifier(X_train_reduced, y_train, C, sigma):
    alpha, b = train_svm(X_train_reduced, y_train, C, sigma)
    return alpha, b

# evaluate trained SVM on test data
def evaluate_model(X_test, y_test, X_train, alpha, b, sigma):
    # decision func for each sample
    K_test = rbf_kernel(X_test, X_train, sigma)
    decision_function = np.dot(K_test, alpha) + b

    predictions = np.sign(decision_function)
    error = np.mean(predictions != y_test.flatten())
    return error

# helper for multiprogramming
def experiment(k, s, X_train, y_train, X_test, y_test, eigenvectors, C, sigma, num_experiments=10):
    num_experiments = 10
    test_errors = []
    for _ in range(num_experiments):
        top_k_eigenvectors = eigenvectors[:, :k]

        pi_j = compute_feature_probabilities(top_k_eigenvectors)
        sampled_features = sample_features(pi_j, s)

        X_train_reduced = reduce_dataset(X_train, sampled_features)
        X_test_reduced = reduce_dataset(X_test, sampled_features)

        alpha, b = train_svm_classifier(X_train_reduced, y_train, C, sigma)
        test_error = evaluate_model(X_test_reduced, y_test, X_train_reduced, alpha, b, sigma)
        test_errors.append(test_error)
    result = np.mean(test_errors)
    return k, s, result

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_process_data()
    eigenvalues, eigenvectors = PCA(X_train)

    k_vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    s_vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    c = 1.0
    sigma = 1.0

    results = Parallel(n_jobs=-1)(delayed(experiment)(k, s, X_train, y_train, X_test, y_test, eigenvectors, c, sigma)
                                  for k in k_vals for s in s_vals)

    # display all results from each averaged experiment
    results_df = pd.DataFrame(results, columns=['k', 's', 'Average Test Error'])
    print(results_df)
    best_result = results_df.loc[results_df['Average Test Error'].idxmin()]
    print("Best Result:")
    print(best_result)



# Third alternative, just SVM, no PCA on same data set

########## ########## ########## ########## ########## ########## ########
########## USED TO JUST RUN SVM, NOT PCA WITH FEATURE SELECTION ##########
########## ########## ########## ########## ########## ########## ########

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers


def load_and_process_data():
    # Load data
    train_data = pd.read_csv("gisette_train.data", header=None).values
    validation_data = pd.read_csv("gisette_valid.data", header=None).values
    test_data = pd.read_csv("gisette_test.data", header=None).values

    train_y = train_data[:, 0].reshape(-1, 1)
    train_y[train_y==0] = -1
    train_X = train_data[:, 1:]
    validation_y = validation_data[:, 0].reshape(-1, 1)
    validation_y[validation_y==0] = -1
    validation_X = validation_data[:, 1:]
    test_y = test_data[:, 0].reshape(-1, 1)
    test_y[test_y==0] = -1
    test_X = test_data[:, 1:]

    # normalize all data
    X_train_norm = normalize(train_X)
    X_val_norm = normalize(validation_X)
    X_test_norm = normalize(test_X)

    return X_train_norm, train_y, X_val_norm, validation_y, X_test_norm, test_y

# zero mean and variance of 1
def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1

    X_norm = (X - mean) / std
    return X_norm

# rbf kernel for use with SVM
def rbf_kernel(X1, X2, sigma):
    sq_dist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-sq_dist / (2 * sigma ** 2))

# train svm wit slack with rbf kernel matrix
def train_svm(X_train, y_train, C, sigma):
    n_samples = X_train.shape[0]

    # compute kernel matrix
    K = rbf_kernel(X_train, X_train, sigma)
    P = matrix(np.outer(y_train, y_train) * K)
    q = matrix(-np.ones((n_samples, 1)))

    # constraints Gx <= h
    G_std = np.diag(-np.ones(n_samples))  # G for alpha >= 0
    G_slack = np.diag(np.ones(n_samples))  # G for alpha <= C
    G = matrix(np.vstack((G_std, G_slack)))
    h = matrix(np.hstack((np.zeros(n_samples), C * np.ones(n_samples))))

    # equality constraint Ax = b
    A = matrix(y_train.T, (1, n_samples), 'd')
    b = matrix(0.0)

    # solve QP problem
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)

    # extract lagrange multipliers
    alpha = np.array(solution['x']).flatten()

    # compute bias term w/ support vectors
    support_indices = (alpha > 1e-5) & (alpha < C - 1e-5)
    y_train = y_train.flatten()
    b = np.mean(y_train[support_indices] - np.dot(K[support_indices], (alpha * y_train).reshape(-1, 1)).flatten())
    return alpha, b

# SVM classifier helper
def train_svm_classifier(X_train_reduced, y_train, C, sigma):
    alpha, b = train_svm(X_train_reduced, y_train, C, sigma)
    return alpha, b

# evaluate trained SVM on test data
def evaluate_model(X_test, y_test, X_train, alpha, b, sigma):
    # decision func for each sample
    K_test = rbf_kernel(X_test, X_train, sigma)
    decision_function = np.dot(K_test, alpha) + b

    predictions = np.sign(decision_function)
    error = np.mean(predictions != y_test.flatten())
    return error

# helper for multiprogramming
def experiment(X_train, y_train, X_test, y_test, C, sigma):
    alpha, b = train_svm_classifier(X_train, y_train, C, sigma)
    test_error = evaluate_model(X_test, y_test, X_train, alpha, b, sigma)
    result = test_error
    return result

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_process_data()

    c = 1.0
    sigma = 1.0

    results = experiment(X_train, y_train, X_test, y_test, c, sigma)

    # display all results from each experiment
    print(results)

# kmeans++ algorithm, which is an alternative implementations of standard k-means algorithm that will choose
# a center from a distribution instead of randomly selecting, resulting in tigher groupings

import numpy as np
import pandas as pd

data = pd.read_csv('leaf.data', header=None).values
labels = data[:, 0]
features = data[:, 1:]

# standardize features (mean zero, variance one)
features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
k_values = [10, 20, 30, 36, 40]

# k-means (random center generation)
def k_means(X, k, max_iters=100):
    # randomly initialize k cluster centers
    centers = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(max_iters):
        # assign points to nearest center
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        # check convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    # calculate k-means objective
    objective = np.sum([np.linalg.norm(X[labels == i] - centers[i]) ** 2 for i in range(k)])
    return objective

# k-means++ center generation
def k_means_plus_plus(X, k):
    centers = [X[np.random.choice(len(X))]]
    for _ in range(1, k):
        dists = np.min([np.linalg.norm(X - c, axis=1) ** 2 for c in centers], axis=0)
        probs = dists / dists.sum()
        new_center = X[np.random.choice(len(X), p=probs)]
        centers.append(new_center)
    return np.array(centers)

# k-means with k-means++ initialization
def k_means_with_plus_plus(X, k, max_iters=100):
    centers = k_means_plus_plus(X, k)
    for _ in range(max_iters):
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    objective = np.sum([np.linalg.norm(X[labels == i] - centers[i]) ** 2 for i in range(k)])
    return objective

# running both methods
def run_experiments(X, k_values, num_runs=100):
    results = {}
    for k in k_values:
        k_means_objectives = [k_means(X, k) for _ in range(num_runs)]
        k_means_plus_plus_objectives = [k_means_with_plus_plus(X, k) for _ in range(num_runs)]

        results[k] = {
            "k-means": (np.mean(k_means_objectives), np.std(k_means_objectives)),
            "k-means++": (np.mean(k_means_plus_plus_objectives), np.std(k_means_plus_plus_objectives))
        }
    return results


results = run_experiments(features, k_values)

# evaluation
for k in k_values:
    mean_kmeans, std_kmeans = results[k]['k-means']
    mean_kmeanspp, std_kmeanspp = results[k]['k-means++']
    print(f"k = {k}")
    print(f"Random Initialization: Mean Objective = {mean_kmeans:.4f}, Std Dev = {std_kmeans:.4f}")
    print(f"k-Means++ Initialization: Mean Objective = {mean_kmeanspp:.4f}, Std Dev = {std_kmeanspp:.4f}")
