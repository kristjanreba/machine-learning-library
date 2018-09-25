import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def normalize(X):
    means = np.mean(X, axis=1)
    std_dev = np.std(X, axis=1)
    X = (X - mean) / std_dev


def normal_equation(X, y):
    X_t = np.transpose(X)
    theta = np.linalg.inv(X_t.dot(X))
    theta = theta.dot(X_t)
    theta = theta.dot(y)
    return theta


def cost(X, y, theta, lambd=0.0):
    m = len(y)
    diff = X.dot(theta) - y
    sum = np.matmul(np.transpose(diff), diff)
    J = 1/(2*m)*sum # + lambd/(2*m)*np.sum(np.square(theta))
    #print('Jc: ', J)
    return J


def cost_derivative(X, y, theta, lambd=0.0):
    m = len(y)
    diff = X.dot(theta) - y
    X_t = np.transpose(X)
    J_d = 1/m * np.matmul(X_t, diff)  # + (lambd/m)*theta
    return J_d


def gradient_descent(X, y, learning_rate, num_iter, lambd=0.0):
    m = len(y)
    J_history = np.zeros(num_iter)
    theta = np.random.randn(2,1)

    for iter in range(num_iter):
        theta -= learning_rate * cost_derivative(X, y, theta)
        J_history[iter] = cost(X, y, theta)

    return theta, J_history


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data['x'].values
y_train = train_data['y'].values
X_test = test_data['x'].values
y_test = test_data['y'].values

X = X_train
y = y_train

m = X.shape[0]
X = np.reshape(X, (m,1))
y = np.reshape(y, (m,1))

bias = np.ones((m,1))
X = np.append(X, bias, axis=1) # add bias to X matrix


learning_rate = 10e-5
num_iter = 10

theta, J_history = gradient_descent(X, y, learning_rate, num_iter)
#theta = normal_equation(X, y)

y_pred = X.dot(theta)


plt.scatter(X[:,0], y, s=3)
plt.plot(X[:,0], y_pred, color='r')
plt.show()


plt.plot(J_history)
plt.show()
