# -*- coding: utf-8 -*-


## You can put your name here

* Name: Kevin Kelly
* Net ID: kak230001



Problem Set 2


# Problem 1: Mesothelioma data


# Problem 1, part 1: Prival SVM
import numpy as np
import pandas as pd
import cvxpy as cp

data = pd.read_csv('meso.data', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1]
y = y.map({1:-1, 2:1}) # map labels to work with cvxpy (1 --> -1, 2 --> 1)
y= y.values # convert to np array

# separate data into train, validate, test segments
X_train, y_train = X[:194], y[:194]
X_val, y_val = X[194:291], y[194:291]
X_test, y_test = X[291:], y[291:]

# define c values
C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

def primal_svm_train(X, y, c):
  M,n = X.shape
  w = cp.Variable(n)
  b = cp.Variable()
  xi = cp.Variable(M)

  # objective function: min(1/2||w||^2 + c*sum(xi)))
  objective = cp.Minimize(0.5 * cp.norm(w, 2)**2 + c * cp.sum(xi))

  # constraints: y(w^T*x^(i) + b) >= 1 - xi, for all i
  # & xi >= 0, for all i
  constraints = [y[i] * (X[i] @ w + b) >= 1 - xi[i] for i in range(M)]
  constraints += [xi >= 0]

  # set up problem
  prob = cp.Problem(objective, constraints)

  # solve problem
  prob.solve()

  return w.value, b.value

# generate predictions based on w and b values and test
def eval_accuracy(X, y, w, b):
  predictions = np.sign(X @ w + b)
  accuracy = np.mean(predictions == y)
  return accuracy

best_val_c = None
best_val_accuracy = 0
best_test_accuracy = 0

# create models for each c value
for c in C_list:
  # traning set
  w, b = primal_svm_train(X_train, y_train, c)

  # test on training
  training_accuracy = eval_accuracy(X_train, y_train, w, b)
  print(f"C={c}, Training Accuracy: {training_accuracy:.4f}")

  # test on validation
  validation_accuracy = eval_accuracy(X_val, y_val, w, b)
  print(f"C={c}, Validation Accuracy: {validation_accuracy:.4f}")

  # test on test
  test_accuracy = eval_accuracy(X_test, y_test, w, b)
  print(f"C={c}, Test Accuracy: {test_accuracy:.4f}")

  if validation_accuracy > best_val_accuracy:
    best_val_accuracy = validation_accuracy
    best_val_c = c
    best_test_accuracy = test_accuracy;

  print()

print(f"Best C = {best_val_c}, Best Test Accuracy = {best_test_accuracy:.4f}")


# Problem 1, part 2: Dual SVM w/ Gaussian Kernel

import numpy as np
import pandas as pd
import cvxpy as cp

def dual_svm_train(X, y, c, sigma):
  M,n = X.shape
  K = kernel_matrix(X, sigma)
  lambdas = cp.Variable(M)

  # set up objective function
  objective = cp.Maximize(cp.sum(lambdas) - 0.5 * cp.quad_form(cp.multiply(lambdas, y), K))

  # define constraints
  constraints = [
    lambdas >= 0,
    lambdas <= c,
    cp.sum(cp.multiply(lambdas, y)) == 0
  ]

  problem = cp.Problem(objective, constraints)
  problem.solve()

  lagrange_multipliers = lambdas.value
  support_vectors = np.where((lagrange_multipliers > 1e-5) & (lagrange_multipliers < c))[0]

  # calculate bias given support vectors
  if len(support_vectors) > 0:
    bias_vals = []
    for sv_i in support_vectors:
      b_i = y[sv_i] -np.sum(lambdas.value * y * K[sv_i])
      bias_vals.append(b_i)
    bias = np.mean(bias_vals)

  return lambdas.value, bias

# gaussian kernel function
def gaussian_kernel(x1, x2, sigma):
  # Compute the Gaussian kernel between two samples
  return np.exp(-np.linalg.norm(x1-x2)**2 / (2 * sigma ** 2))

# compute kernel matrix
def kernel_matrix(X, sigma):
  M = X.shape[0]
  K = np.zeros((M,M))
  for i in range(M):
    for j in range(M):
      K[i,j] = gaussian_kernel(X[i], X[j], sigma)
  return K

# Compute the accuracy of the predictions
def compute_accuracy(predictions, y_true):
  return sum(p == t for p, t in zip(predictions, y_true)) / len(y_true)

# predict with gaussian kernel
def predict(lambdas, X_train, y_train, X_test, sigma, bias):
  predictions = []
  for x in X_test:
    weighted_sum = 0
    for i in range(len(X_train)):
      weighted_sum += lambdas[i] * y_train[i] * gaussian_kernel(x, X_train[i], sigma)
    predictions.append(np.sign(weighted_sum + bias))
  return np.array(predictions)

def main():
  data = pd.read_csv('meso.data', header=None)
  X = data.iloc[:, :-1].values
  y = data.iloc[:, -1]
  y = y.map({1:-1, 2:1}) # map labels to work with cvxpy (1 --> -1, 2 --> 1)
  y= y.values # convert to np array

  # separate data into train, validate, test segments
  X_train, y_train = X[:194], y[:194]
  X_val, y_val = X[194:291], y[194:291]
  X_test, y_test = X[291:], y[291:]

  c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
  sigma_values = [0.001, 0.01, 0.1, 1, 10, 100]

  best_c = None
  best_sigma = None
  best_lambdas = None
  best_validation_acc = 0

  for i in range(len(c_values)):
    for j in range(len(sigma_values)):
      lambdas, bias = dual_svm_train(X_train, y_train, c_values[i], sigma_values[j])

      train_acc = compute_accuracy(predict(lambdas, X_train, y_train, X_train, sigma_values[j], bias), y_train)
      print(f"C: {c_values[i]}, sigma: {sigma_values[j]}, train accuracy: {train_acc}")
      val_acc = compute_accuracy(predict(lambdas, X_train, y_train, X_val, sigma_values[j], bias), y_val)
      print(f"C: {c_values[i]}, sigma: {sigma_values[j]}, validation accuracy: {val_acc}")

      if val_acc > best_validation_acc:
        best_validation_acc = val_acc
        best_c = c_values[i]
        best_sigma = sigma_values[j]
        best_lambdas = lambdas
        best_bias = bias

  print(f"Best C: {best_c}")
  print(f"Best sigma: {best_sigma}")
  print(f"Best validation accuracy: {best_validation_acc}")

  test_acc = compute_accuracy(predict(best_lambdas, X_train, y_train, X_test, best_sigma, best_bias), y_test)
  print(f"Test accuracy: {test_acc}")

if __name__ == "__main__":
  main()


# Problem 3: Poisonous Musroom dataset with decison trees

import numpy as np
import pandas as pd

# TreeNode class to represent each node in the decision tree
# Attributes: name, attribute_index, level, answer, child_nodes, parent_type, majority
class TreeNode:
  def __init__(self, name=None, attribute_index=None, level=0, answer=None, child_nodes=None, parent_type=None, majority=None, info_gain=None):
    self.name = name
    self.attribute_index = attribute_index
    self.level = level
    self.answer = answer
    self.child_nodes = child_nodes
    self.parent_type = parent_type
    self.majority = majority
    self.info_gain = info_gain # added per TA's response in piazza

# Function to calculate parent entropy based on class distribution
# Returns: total entropy, predicted class if pure, and majority class
def calculate_entropy(y):
  label_counts = np.unique(y, return_counts=True)[1]
  total_samples = len(y)
  entropy = 0
  for count in label_counts:
    probability = count / total_samples # probability of each label
    entropy += -probability * np.log2(probability) # formula
  majority_class = np.argmax(label_counts) # class with highest count
  return entropy, majority_class

# Function to calculate child entropy for a given attribute
# Splits data based on unique attribute values and computes entropy
# Returns: total weighted child entropy
def child_entropy(data, attribute_index):
  y = data[:, 0]
  total_samples = len(y)
  attribute_values = data[:, attribute_index]
  unique_values = np.unique(attribute_values)
  total_entropy = 0
  for value in unique_values:
    subset = data[attribute_values == value] # subset of data with specific attribute value
    labels = subset[:, 0]
    entropy, _ = calculate_entropy(labels) # calculate entropy for subset
    total_entropy += (len(subset) / total_samples) * entropy # weighted entropy
  return total_entropy

# Function to calculate information gain for all attributes
# Finds attribute with highest information gain
# Returns: best attribute index, information gain, predicted class if pure, majority class
def information_gain(data):
  X, y = data[:, 1:], data[:, 0]
  parent_entropy, majority_class = calculate_entropy(y)
  best_gain = -1
  best_attribute = -1
  for i in range(1, X.shape[1]): # iterate over all attributes
    gain = parent_entropy - child_entropy(data, i) # info gain for attribute
    if gain > best_gain: # update best if current is higher
      best_gain = gain
      best_attribute = i
    elif gain == best_gain:
      best_attribute = max(best_attribute, i)
  return best_attribute, best_gain, majority_class

# Function to partition the data based on the attribute selected for splitting
# Returns: list of partitions of the dataset
def _partition(data, attribute_index):
  attribute_values = data[:, attribute_index]
  unique_values = np.unique(attribute_values)
  partitions = {value: data[attribute_values == value] for value in unique_values} # create subsets for specific attribute
  return partitions

# Recursive function to build the decision tree
# Creates a TreeNode for the best split and recursively partitions data until no gain
# Stops if all data is pure or max depth is reached
def build_tree(data, level=0):
  y = data[:, 0]
  unique_labels = np.unique(y)

  # all data has same class level, return leaf node w/ the class
  if len(unique_labels) == 1:
    return TreeNode(name="Leaf", level=level, answer=unique_labels[0])

  best_attribute, best_gain, majority_class = information_gain(data)

  # no information gain possible, so return leaf node w/ majority class
  if best_gain == 0:
    return TreeNode(name="Leaf", level=level, answer=majority_class)

  # partition based on best attribute
  partitions = _partition(data, best_attribute)
  child_nodes = {}

  # recursively build child nodes for each partition
  for value, partition in partitions.items():
    child_nodes[value] = build_tree(partition, level + 1)

  return TreeNode(name=f"Attribute {best_attribute}", attribute_index=best_attribute, level=level, child_nodes=child_nodes, majority=majority_class, info_gain=best_gain)

# Function to predict class labels using the decision tree
# Traverses the tree based on the attribute values of the input row
# Returns: predicted class label or majority class if at max depth
def predict(tree, row):
  if tree.answer is not None: # leaf, so return class label
    return tree.answer
  attribute_value = row[tree.attribute_index]

  # keep working down tree until child is found. if not then return majority
  if attribute_value in tree.child_nodes:
    return predict(tree.child_nodes[attribute_value], row)
  else:
    return tree.majority

# used to help draw tree as part of this question
def print_tree(node, spacing=""):
    # if it's a leaf node, print the class label
    if node.answer is not None:
        print(spacing + f"Leaf: Predict {node.answer}")
        return

    # print the decision at the current node
    print(spacing + f"Node: Attribute {node.attribute_index} (Info Gain: {node.info_gain:.4f})")

    # print the branches for each value of the attribute
    for value, child in node.child_nodes.items():
        print(spacing + f"--> Value {value}:")
        print_tree(child, spacing + "  ")


# Main flow: Read training and test data, build the tree using recursion,
# and calculate accuracy of predictions on the test data.
def main():
  # load data
  train_data = pd.read_csv("mush_train.data").values
  test_data = pd.read_csv("mush_test.data").values

  # build tree
  tree = build_tree(train_data)
  print("Decision Tree:")
  print_tree(tree)

  # evaluate tree
  correct = 0
  for row in test_data:
    prediction = predict(tree, row)
    if prediction == row[0]:
      correct += 1
  accuracy = correct / len(test_data)
  print("Accuracy:", accuracy)

if __name__ == "__main__":
  main()

