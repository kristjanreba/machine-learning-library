import numpy as np
import matplotlib.pyplot as plt


def normalize(X):
    means = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    X = (X - means) / std_dev
    return X


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
    J = 1/(2*m)*sum
    J += lambd/(2*m)*np.sum(np.square(theta))
    return J


def cost_derivative(X, y, theta, lambd=0.0):
    m = len(y)
    diff = X.dot(theta) - y
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


def plot_regression_line(X, y, theta, y_pred, degree):
    xmin = np.min(X[:,0])
    xmax = np.max(X[:,0])

    X_plot = np.linspace(xmin, xmax, 200)
    m = X_plot.shape[0]
    X_plot = np.reshape(X_plot, (m,1))

    bias = np.ones((m,1))
    X_plot = np.append(X_plot, bias, axis=1)

    X_plot = add_poly_features(X_plot, degree)
    y_plot = X_plot.dot(theta)


    plt.scatter(X[:,0], y, s=3)
    plt.plot(X_plot[:,0], y_plot, color='r')
    plt.show()


def get_num_features(X):
    try: return X.shape[1]
    except IndexError: return 1



# data we use to test this algorithm
X_train = np.transpose([-1, 0, 1, 2, 3, 5, 7, 9])
y_train = np.transpose([-1, 3, 2.5, 5, 4, 2, 5, 4])

X = X_train
y = y_train

# reshape X and y to be correct dimensions for further work
m = X.shape[0]
num_features = get_num_features(X)
X = np.reshape(X, (m,num_features))
y = np.reshape(y, (m,1))

# normalize the data for faster convergence when using gradient descent
X = normalize(X)

# add bias to X matrix
bias = np.ones((m,1))
X = np.append(X, bias, axis=1)


# hyper-parameters of model
# if using normal_equation you don't need learning_rate, num_iter or lambd
learning_rate = 10e-3 # size of a step in gradient descent
num_iter = 1000 # number of iterations of gradient descent
lambd = 0.01 # regularization parameter
degree = 6 # degree of polynom used to fit the data (degree = 1 -> linear regression)


X = add_poly_features(X, degree)

# uncomment the algorithm you want for fitting the function
theta, J_history = gradient_descent(X, y, learning_rate, num_iter, lambd)
#theta = normal_equation(X, y)

# make predictions with this linear model
y_pred = X.dot(theta)

plot_regression_line(X, y, theta, y_pred, degree)


# if using normal_equation comment out this 2 lines
# because J_history is only produced using gradient descent
plt.plot(J_history)
plt.show()
