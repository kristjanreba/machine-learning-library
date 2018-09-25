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
    num_features = X.shape[1]
    J_history = np.zeros(num_iter)
    theta = np.random.randn(num_features,1)

    for iter in range(num_iter):
        theta -= learning_rate * cost_derivative(X, y, theta)
        J_history[iter] = cost(X, y, theta)

    return theta, J_history


def add_poly_features(X, degree):
    m = X.shape[0]
    for n in range(2, degree+1):
        new_feature = np.power(X[:,0], n)
        new_feature = np.reshape(new_feature, (m,1))
        X = np.append(X, new_feature, axis=1)
    return X


X_train = np.transpose([-1, 0, 1, 2, 3, 5, 7, 9])
y_train = np.transpose([-1, 3, 2.5, 5, 4, 2, 5, 4])

X = X_train
y = y_train

m = X.shape[0]
X = np.reshape(X, (m,1))
y = np.reshape(y, (m,1))

bias = np.ones((m,1))
X = np.append(X, bias, axis=1) # add bias to X matrix

degree = 2
X = add_poly_features(X, degree)




#learning_rate = 10e-9
#num_iter = 10000

#theta, J_history = gradient_descent(X, y, learning_rate, num_iter)
theta = normal_equation(X, y)

y_pred = X.dot(theta)

#X_plot = np.linspace(x_min, x_max, 200)
#y_plot = X_plot.dot(theta)


def plot_regression_line(X, y, y_pred):

    X_plot = np.linspace(-2, 10, 200)
    m = X_plot.shape[0]

    X_plot = np.reshape(X_plot, (m,1))

    bias = np.ones((m,1))
    X_plot = np.append(X_plot, bias, axis=1)
    X_plot = add_poly_features(X_plot, degree)
    y_plot = X_plot.dot(theta)


    plt.scatter(X[:,0], y, s=3)
    plt.plot(X_plot[:,0], y_plot, color='r')
    plt.show()


plt.plot(J_history)
plt.show()
