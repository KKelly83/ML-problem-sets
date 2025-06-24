# Logistic regression with sonar data set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt


data_train = pd.read_csv('sonar_train.data', header=None)
data_valid = pd.read_csv('sonar_valid.data', header=None)
data_test = pd.read_csv('sonar_test.data', header=None)

X_train = data_train.iloc[:, :-1].values
y_train = data_train.iloc[:, -1].replace(1, 0).replace(2, 1).values
X_valid = data_valid.iloc[:, :-1].values
y_valid = data_valid.iloc[:, -1].replace(1, 0).replace(2, 1).values
X_test = data_test.iloc[:, :-1].values
y_test = data_test.iloc[:, -1].replace(1, 0).replace(2, 1).values


class LogisticRegression:
  def __init__(self, lr=0.1, decay=0.01, n_iters=10000):
    self.lr = lr
    self.decay = decay
    self.n_iters = n_iters
    self.weights = None
    self.bias = None

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def compute_loss(self, y_pred, y):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    for _ in range(self.n_iters):
      # update lr
      lr = self.lr / (1 + self.decay * _)

      # compute predictions
      linear_pred = np.dot(X, self.weights) + self.bias
      y_pred = self.sigmoid(linear_pred)

      # compute gradients
      dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
      db = (1 / n_samples) * np.sum(y_pred - y)

      # update params
      self.weights -= self.lr * dw
      self.bias -= self.lr * db

  def predict(self, X):
    linear_pred = np.dot(X, self.weights) + self.bias
    y_pred = self.sigmoid(linear_pred)
    return np.where(y_pred > 0.5, 1, 0)

class LogisticRegressionL2:
  def __init__(self, lr=0.1, decay=0.01, n_iters=10000, reg_const=0.01):
    self.lr = lr
    self.decay = decay
    self.n_iters = n_iters
    self.reg_const = reg_const
    self.weights = None
    self.bias = None

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def compute_loss(self, y_pred, y):
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    l2_penalty = (self.reg_const / 2) * np.sum(self.weights ** 2)
    return loss + l2_penalty

  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    for _ in range(self.n_iters):
      # update lr
      lr = self.lr / (1 + self.decay * _)

      # compute predictions
      linear_pred = np.dot(X, self.weights) + self.bias
      y_pred = self.sigmoid(linear_pred)

      # compute gradients
      dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + self.reg_const * self.weights
      db = (1 / n_samples) * np.sum(y_pred - y)

      # update params
      self.weights -= self.lr * dw
      self.bias -= self.lr * db

  def predict(self, X):
    linear_pred = np.dot(X, self.weights) + self.bias
    y_pred = self.sigmoid(linear_pred)
    return np.where(y_pred > 0.5, 1, 0)

class LogisticRegressionL1:
    def __init__(self, lr=0.1, decay=0.01, n_iters=10000, reg_const=0.01):
        self.lr = lr
        self.decay = decay
        self.n_iters = n_iters
        self.reg_const = reg_const
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, y_pred, y):
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        l1_penalty = self.reg_const * np.sum(np.abs(self.weights))
        return loss + l1_penalty

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            lr = self.lr / (1 + self.decay * i)
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + self.reg_const * np.sign(self.weights)
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= lr * dw
            self.bias -= lr * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        return np.where(y_pred > 0.5, 1, 0)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

pred_train = log_reg.predict(X_train)
print(f'Normal Logistic Regression Train Accuracy: {np.mean(pred_train == y_train)}')

pred_test = log_reg.predict(X_test)
print(f'Normal Logistic Regression Test Accuracy: {np.mean(pred_test == y_test)}')
print()

l2_best_lambda = None
l2_best_accuracy = 0
l2_best_weighjts = None
l2_best_bias = None

for reg_const in [0.001, 0.01, 0.1, 1, 10]:
  log_reg = LogisticRegressionL2(reg_const=reg_const)
  log_reg.fit(X_train, y_train)

  pred_valid = log_reg.predict(X_valid)
  accuracy = np.mean(pred_valid == y_valid)

  if accuracy > l2_best_accuracy:
    l2_best_accuracy = accuracy
    l2_best_lambda = reg_const
    l2_best_weights = log_reg.weights
    l2_best_bias = log_reg.bias

l2_best_model = LogisticRegressionL2(reg_const=l2_best_lambda)
l2_best_model.fit(X_train, y_train)
l2_pred_test = l2_best_model.predict(X_test)
l2_test_accuracy = np.mean(l2_pred_test == y_test)
print("L2")
print(f'Selected lambda: {l2_best_lambda}')
print(f'Learned weights: {l2_best_weights}')
print(f'Learned bias: {l2_best_bias}')
print(f'L2 Logistic Regression Test Accuracy: {l2_test_accuracy}')
print()

l1_best_lambda = None
l1_best_accuracy = 0
l1_best_weights = None
l1_best_bias = None

for reg_const in [0.001, 0.01, 0.1, 1, 10]:
    log_reg_l1 = LogisticRegressionL1(reg_const=reg_const)
    log_reg_l1.fit(X_train, y_train)

    pred_valid = log_reg_l1.predict(X_valid)
    accuracy = np.mean(pred_valid == y_valid)

    if accuracy > l1_best_accuracy:
        l1_best_accuracy = accuracy
        l1_best_lambda = reg_const
        l1_best_weights = log_reg_l1.weights
        l1_best_bias = log_reg_l1.bias

# evaluate on the test set
l1_best_model = LogisticRegressionL1(reg_const=l1_best_lambda)
l1_best_model.fit(X_train, y_train)
l1_pred_test = l1_best_model.predict(X_test)
l1_test_accuracy = np.mean(l1_pred_test == y_test)

print("L1")
print(f'Selected λ: {l1_best_lambda}')
print(f'Learned weights: {l1_best_weights}')
print(f'Learned bias: {l1_best_bias}')
print(f'L1 Logistic Regression Test Accuracy: {l1_test_accuracy:.4f}')

print()

def generate_data(n_samples=100, separation=0.0, noise=0.5):
    X = np.random.randn(n_samples, 2) * noise
    y = np.where(X[:, 0] + X[:, 1] > separation, 1, 0)
    print(f"Class 0 count: {np.sum(y == 0)}")
    print(f"Class 1 count: {np.sum(y == 1)}")
    return X, y



X_gen, y_gen = generate_data(200)
y_gen_svm = y_gen * 2 - 1  # adjust labels to 1,-1 for SVM

log_reg = LogisticRegression(lr=0.1, decay=0.01, n_iters=1000)  # without regularization
log_reg.fit(X_gen, y_gen)
print("Weights without regularization:", log_reg.weights)

# split into sets
X_train, X_valid, X_test = X_gen[:160], X_gen[160:180], X_gen[180:]
y_train, y_valid, y_test = y_gen[:160], y_gen[160:180], y_gen[180:]

def train_svm(X, y):
    n_samples, n_features = X.shape
    K = np.dot(X, X.T)
    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-np.ones(n_samples))
    G = cvxopt.matrix(-np.eye(n_samples))
    h = cvxopt.matrix(np.zeros(n_samples))
    A = cvxopt.matrix(y, (1, n_samples), 'd')
    b = cvxopt.matrix(0.0)

    # solve QP problem
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.array(solution['x']).flatten()

    # find support vectors
    sv = alphas > 1e-5
    w = np.sum((alphas * y)[:, np.newaxis] * X, axis=0)
    b = np.mean(y[sv] - np.dot(X[sv], w))

    return w, b

# plot Decision Boundary
def plot_decision_boundary(w, b, label, color):
    x_points = np.linspace(-2, 2, 100)
    y_points = -(w[0] * x_points + b) / w[1]
    plt.plot(x_points, y_points, label=label, color=color)

# train the Logistic Regression L2 model
l2_best_lambda = None
l2_best_accuracy = 0
l2_best_weights = None
l2_best_bias = None

for reg_const in [0.001, 0.01, 0.1, 1, 10]:
    log_reg_l2 = LogisticRegressionL2(reg_const=reg_const)
    log_reg_l2.fit(X_train, y_train)

    pred_valid = log_reg_l2.predict(X_valid)
    accuracy = np.mean(pred_valid == y_valid)

    if accuracy > l2_best_accuracy:
        l2_best_accuracy = accuracy
        l2_best_lambda = reg_const
        l2_best_weights = log_reg_l2.weights
        l2_best_bias = log_reg_l2.bias

l2_best_model = LogisticRegressionL2(reg_const=l2_best_lambda)
l2_best_model.fit(X_train, y_train)
w_l2, b_l2 = l2_best_model.weights, l2_best_model.bias

# train the SVM model
w_svm, b_svm = train_svm(X_gen, y_gen_svm)

plt.figure(figsize=(10, 6))
plt.scatter(X_gen[y_gen == 0][:, 0], X_gen[y_gen == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X_gen[y_gen == 1][:, 0], X_gen[y_gen == 1][:, 1], color='red', label='Class 1')

# plot decision boundaries
plot_decision_boundary(w_l2, b_l2, 'Logistic Regression L2', 'green')
plot_decision_boundary(w_svm, b_svm, 'SVM', 'purple')

plt.legend()
plt.title('Decision Boundaries: SVM vs Logistic Regression (L2)')
plt.show()

# Gaussian Naiive Bayes solution with sonar data set

import numpy as np
import pandas as pd


data_train = pd.read_csv('sonar_train.data', header=None)
data_valid = pd.read_csv('sonar_valid.data', header=None)
data_test = pd.read_csv('sonar_test.data', header=None)

X_train = data_train.iloc[:, :-1].values
y_train = data_train.iloc[:, -1].replace(1, 0).replace(2, 1).values
X_valid = data_valid.iloc[:, :-1].values
y_valid = data_valid.iloc[:, -1].replace(1, 0).replace(2, 1).values
X_test = data_test.iloc[:, :-1].values
y_test = data_test.iloc[:, -1].replace(1, 0).replace(2, 1).values

class GaussianNB:
  def fit(self, X, y):
    self.classes = np.unique(y)
    self.mean = {}
    self.var = {}
    self.prior = {}

    for c in self.classes:
      X_c = X[y == c]
      self.mean[c] = np.mean(X_c, axis=0)
      self.var[c] = np.var(X_c, axis=0)
      self.prior[c] = X_c.shape[0] / X.shape[0]

  def predict(self, X):
    posteriors = []
    for x in X:
      posterior = {}
      for c in self.classes:
        prior = np.log(self.prior[c])
        likelihood = np.sum(-0.5 * np.log(2 * np.pi * self.var[c]) - 0.5 * ((x - self.mean[c])**2) / self.var[c])
        posterior[c] = prior + likelihood
      posteriors.append(max(posterior, key=posterior.get))
    return np.array(posteriors)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy}')


# Gaussian mixtures vs k-means implementations with leaf data set

import numpy as np
import pandas as pd

data = pd.read_csv('leaf.data', header=None).values
labels = data[:, 0]
features = data[:, 1:]

# standardize features (mean zero, variance one)
features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
k_values = [10, 20, 30, 36]

epsilon = 1e-6 # eivenvalue threshold
num_initializations = 20

def kmeanspp(features, k):
  n_samples, n_features = features.shape
  centers = np.zeros((k, n_features))
  centers[0] = features[np.random.randint(n_samples)]

  for i in range(1, k):
    distances = np.min(np.linalg.norm(features[:, np.newaxis] - centers[:i], axis=2), axis=1)
    probabilities = distances ** 2 / np.sum(distances ** 2)
    centers[i] = features[np.random.choice(n_samples, p=probabilities)]

  return centers

def gmm(features, k, epsilon, max_iterations=1000, tol=1e-4):
  n_samples, n_features = features.shape
  means = kmeanspp(features, k)
  covariances = np.array([np.eye(n_features)] * k)
  weights = np.ones(k) / k
  log_likelihoods = []

  for i in range(max_iterations):
    # E step
    responsibilities = np.zeros((n_samples, k))
    for j in range(k):
        cov = covariances[j]
        mean = means[j]
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        norm_const = 1 / np.sqrt((2 * np.pi) ** n_features * max(det_cov, epsilon))
        diff = features - mean
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        responsibilities[:, j] = norm_const * np.exp(exponent)

    log_likelihood = np.sum(np.log(np.sum(responsibilities * weights, axis=1) + epsilon))
    responsibilities = responsibilities * weights
    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

    # M step
    Nk = np.sum(responsibilities, axis=0)
    means = np.dot(responsibilities.T, features) / Nk[:, np.newaxis]
    weights = Nk / n_samples

    for j in range(k):
      diff = features - means[j]
      covariances[j] = np.dot((responsibilities[:, j][:, np.newaxis] * diff).T, diff) / Nk[j]
      # correct eigenvalues
      eigenvalues, eigenvectors = np.linalg.eigh(covariances[j])
      eigenvalues[eigenvalues < epsilon] = epsilon
      covariances[j] = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # log likelihood
    log_likelihood = np.sum(np.log(np.sum(responsibilities * weights, axis=1)))
    log_likelihoods.append(log_likelihood)

    if i > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
      break

  return log_likelihoods[-1]

results = {k: [] for k in k_values}

for k in k_values:
  for i in range(num_initializations):
    log_likelihood = gmm(features, k, epsilon)
    results[k].append(log_likelihood)
  mean_log_likelihood = np.mean(results[k])
  variance_log_likelihood = np.var(results[k])
  print(f"k={k}, mean log likelihood={mean_log_likelihood}, variance={variance_log_likelihood}")
