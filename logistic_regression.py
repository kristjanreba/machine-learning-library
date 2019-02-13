import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    g = sigmoid(x)
    return g * (1-g)

def cost_logistic(X, y, theta, lambd=0.0):
    z = X.dot(theta)
    h = sigmoid(z)
    J = 1/m * (-np.transpose(y) * np.log(h) - np.transpose(1-y) * np.log(1-h))
    J += lambd/(2*m)*np.sum(np.square(theta))

def cost_logistic_derivative(X, y, theta, lambd=0.0):
    m = len(y)
    diff = sigmoid(X.dot(theta)) - y
    X_t = np.transpose(X)
    J_d = 1/m * np.matmul(X_t, diff)
    J_d += (lambd/m)*theta
    return J_d

def gradient_descent(X, y, learning_rate, num_iter, lambd=0.0):
    m = len(y)
    num_features = X.shape[1]
    J_history = np.zeros(num_iter)
    theta = np.random.randn(num_features,1)

    for iter in range(num_iter):
        theta -= learning_rate * cost_logistic_derivative(X, y, theta)
        J_history[iter] = cost_logistic(X, y, theta)

    return theta, J_history

def predict(X, theta):
    return sigmoid(X.dot(theta))

def get_num_features(X):
    try: return X.shape[1]
    except IndexError: return 1


X, y = make_blobs(n_features=2, centers=2)

m = X.shape[0]
num_features = get_num_features(X)
X = np.reshape(X, (m,num_features))
y = np.reshape(y, (m,1))

bias = np.ones((m,1))
X = np.append(X, bias, axis=1) # add bias to X matrix


learning_rate = 10e-5
num_iter = 10
lambd = 0.01
theta, J_history = gradient_descent(X, y, learning_rate, num_iter, lambd)


y_pred = predict(X, theta)
