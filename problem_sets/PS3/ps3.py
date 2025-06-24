
# Problem Set 3

# Heart data problem with adaboost algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv('heart_train.data', header=None)
data_test = pd.read_csv('heart_test.data', header=None)

train_X = data_train.iloc[:, 1:].values
train_y = data_train.iloc[:, 0].replace(0, -1).values

test_X = data_test.iloc[:, 1:].values
test_y = data_test.iloc[:, 0].replace(0,-1).values

# Parameters
T = 10 # number of rounds

# Variables
training_accuracies = []
testing_accuracies = []

class DecisionTree: # Single decision
    def __init__(self):
        self.polarity = 1 # 1 for positive, -1 for negative
        self.feature_idx = None
        self.split = None
        self.alpha = None
        self.error =None

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        feature_column = X[:, self.feature_idx]

        if self.polarity == 1:
            predictions[feature_column < self.split] = -1
        else:
            predictions[feature_column > self.split] = -1

        return predictions

class AdaBoost:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_trees):
            tree = DecisionTree()
            min_error = float('inf')

            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                splits = np.unique(X_column)

                for split in splits:
                    predictions = np.ones(n_samples)
                    predictions[X_column < split] = -1

                    error = sum(w[y != predictions])

                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    else:
                        p = 1

                    if error < min_error:
                        tree.polarity = p
                        tree.split = split
                        tree.feature_idx = feature_i
                        tree.error = error
                        min_error = error

            #EPS = 1e-10
            tree.alpha = 0.5 * np.log((1.0 - min_error) / (min_error ))
            predictions = tree.predict(X)
            w *= np.exp(-tree.alpha * y * predictions)
            w /= np.sum(w)
            self.trees.append(tree)

            train_predictions = self.predict(train_X)
            train_accuracy = np.mean(train_y == train_predictions)
            training_accuracies.append(train_accuracy)

            test_predictions = self.predict(test_X)
            test_accuracy = np.mean(test_y == test_predictions)
            testing_accuracies.append(test_accuracy)

            print(f"Round {_ + 1}:")
            print(f"Selected tree: Feature {tree.feature_idx}")
            print(f"Alpha: {tree.alpha}")
            print(f"Error: {tree.error}")
            print(f"Threshold: {tree.split}")
            print(f"Training accuracy: {train_accuracy}")
            print(f"Testing accuracy: {test_accuracy}")

    def predict(self, X):
        classifier_preds = [tree.alpha * tree.predict(X) for tree in self.trees]
        y_pred = np.sum(classifier_preds, axis=0)
        return np.sign(y_pred)


ada_boost = AdaBoost(n_trees=T)
ada_boost.fit(train_X, train_y)

# Plot Accuracy:
# - Plot the training accuracy after each round.
# - Plot the testing accuracy after each round.
plt.plot(range(1, T + 1), training_accuracies, label='Training Accuracy')
plt.plot(range(1, T + 1), testing_accuracies, label='Testing Accuracy')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Coordinate descent to minimize exponential loss

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv('heart_train.data', header=None)
data_test = pd.read_csv('heart_test.data', header=None)

train_X = data_train.iloc[:, 1:].values
train_y = data_train.iloc[:, 0].replace(0, -1).values

test_X = data_test.iloc[:, 1:].values
test_y = data_test.iloc[:, 0].replace(0,-1).values

# parameters
M = train_X.shape[0] # number of training samples
N = test_X.shape[0] # number of testing samples

class DecisionTree: # single decision
    def __init__(self):
        self.polarity = 1 # 1 for positive, -1 for negative
        self.feature_idx = None
        self.split = None
        self.error =None

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        feature_column = X[:, self.feature_idx]

        if self.polarity == 1:
            predictions[feature_column < self.split] = -1
        else:
            predictions[feature_column > self.split] = -1

        return predictions

# generate all trees
def generate_trees(X, y):
  n_features = X.shape[1]
  trees = []
  for feature_idx in range(n_features):
    X_column = X[:, feature_idx]
    unique_values = np.unique(X_column)
    for split in unique_values:
      for polarity in [1, -1]: # also split on possible leaf orientations
        tree = DecisionTree()
        tree.feature_idx = feature_idx
        tree.split = split
        tree.polarity = polarity
        trees.append(tree)
  return trees

# generate all trees
trees = generate_trees(train_X, train_y)
n_trees = len(trees)
# predictions for each tree
predictions = np.array([tree.predict(train_X) for tree in trees])

alphas = np.zeros(n_trees)

# conts/variables
max_iters = 100 # arbitrary # of iterations
convergance_threshold = 1e-6
converged = False
iteration = 0

# from slide 21, full exponential loss formula
def exponential_loss(y, f):
  return np.sum(np.exp(-y * f))

# coordinate descent loop
while not converged and iteration < max_iters:
  # for use in determining convergence
  previous_loss = exponential_loss(train_y, np.dot(alphas, predictions))

  # iterate over all trees (alphas)
  for i in range(n_trees):
    margin = np.dot(alphas, predictions) - alphas[i] * predictions[i]

    positive_predictions = predictions[i] == train_y  # all matching predictions
    negative_predictions = ~positive_predictions  # all non-matching predictions

    # numerator
    positive_loss = np.sum(np.exp(-train_y[positive_predictions] * margin[positive_predictions]))
    # denominator
    negative_loss = np.sum(np.exp(-train_y[negative_predictions] * margin[negative_predictions]))

    # a_t' update
    alphas[i] = 0.5 * np.log(positive_loss/negative_loss)

  # calculate loss
  current_loss = exponential_loss(train_y, np.dot(alphas, predictions))

  # determine convergence, otherwise keep going
  if np.abs(current_loss - previous_loss) < convergance_threshold:
    converged = True

  iteration += 1

print("Alphas values:", alphas)
print("Largest alpha:", np.max(alphas))
print("Loss:", current_loss)

def compute_accuracy(X, y, alphas, trees):
    # re-calculate since using different X (train vs test)
    predictions = np.array([tree.predict(X) for tree in trees])

    # weighted sum for each sample w/ alpha
    weighted_sum = np.dot(alphas, predictions)

    # grab sign for prediction
    final_predictions = np.sign(weighted_sum)

    accuracy = np.mean(final_predictions == y)
    return accuracy

# accuracies
train_accuracy = compute_accuracy(train_X, train_y, alphas, trees)
print("Training Accuracy:", train_accuracy)

test_accuracy = compute_accuracy(test_X, test_y, alphas, trees)
print("Test Accuracy:", test_accuracy)

# Bagging with 20 bootstrap samples

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv('heart_train.data', header=None)
data_test = pd.read_csv('heart_test.data', header=None)

train_X = data_train.iloc[:, 1:].values
train_y = data_train.iloc[:, 0].replace(0, -1).values

test_X = data_test.iloc[:, 1:].values
test_y = data_test.iloc[:, 0].replace(0,-1).values

# Parameters
M = train_X.shape[0] # number of training samples
N = test_X.shape[0] # number of testing samples

class DecisionTree: # Single decision
    def __init__(self):
        self.polarity = 1 # 1 for positive, -1 for negative
        self.feature_idx = None
        self.split = None
        self.error =None

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        feature_column = X[:, self.feature_idx]

        if self.polarity == 1:
            predictions[feature_column < self.split] = -1
        else:
            predictions[feature_column > self.split] = -1

        return predictions

# generate all trees
def generate_trees(X, y):
    n_features = X.shape[1]
    trees = []
    for feature_idx in range(n_features):
        unique_values = np.unique(X[:, feature_idx])
        for split in unique_values:
            for polarity in [1, -1]: # also split on possible leaf orientations
                tree = DecisionTree()
                tree.feature_idx = feature_idx
                tree.split = split
                tree.polarity = polarity
                trees.append(tree)
    return trees

# consts/variables for bagging loop
b_samples = 20
selected_trees = []
cumulative_train_predictions = np.zeros(M)
cumulative_test_predicitons = np.zeros(N)

for i in range(b_samples):
    # get bootstrap sample
    indices = np.random.choice(range(len(train_X)), size = len(train_X), replace=True)
    bootstrap_X, bootstrap_y = train_X[indices], train_y[indices]
    trees = generate_trees(bootstrap_X, bootstrap_y)

    # evaluate each tree on the bootstrap sample
    lowest_error = float('inf')
    for tree in trees:
        predictions = tree.predict(bootstrap_X)
        error = np.sum(predictions != bootstrap_y) / len(bootstrap_y)
        if error < lowest_error:
            lowest_error = error
            selected_tree = tree

    # add to bagging model
    selected_trees.append(selected_tree)

    # update with new trees predictions
    cumulative_train_predictions += selected_tree.predict(train_X)
    cumulative_test_predicitons += selected_tree.predict(test_X)
    train_accuracy = np.mean(np.sign(cumulative_train_predictions) == train_y)
    test_accuracy = np.mean(np.sign(cumulative_test_predicitons) == test_y)

    print("Training Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)

# calculate accuracies from cumulative predictions
train_accuracy = np.mean(np.sign(cumulative_train_predictions) == train_y)
test_accuracy = np.mean(np.sign(cumulative_test_predicitons) == test_y)

print()
print("Final Training Accuracy:", train_accuracy)
print("Final Testing Accuracy:", test_accuracy)
