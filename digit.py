import numpy as np
import pandas as pd
import math
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.image import imread


def sigmoid(x):
    g = 1/(1+np.exp(-x))
    return g


# x is m*(n+1) matrix with x[i][0]=1   #theta is (n+1)*1 matrix   #y is m*1 matrix
def cost_fn(x, y, theta):
    m = len(x)
    h = sigmoid(x.dot(theta))  # h is m*1 matrix now
    cost = -(y.T).dot(np.log(h)) - \
        ((1-y).T).dot(np.log(1-h))  # cost is 1*1 matrix
    return cost/m


def gradient_descent(x, y, theta, iters, learning_rate=1):
    m = len(x)
    for k in range(iters):
        h = sigmoid(x.dot(theta))
        new = theta - (learning_rate/m)*(np.transpose(x)).dot(h-y)
        theta = new
    return theta


def predict(x, y, theta, iters, learning_rate=1):
    theta = gradient_descent(x, y, theta, iters, learning_rate=1)
    return sigmoid(x.dot(theta))


# Loading image data from X.csv and coressponding labels from Y.csv
X = pd.read_csv("X.csv", header=None)
X = np.array(X)
y = pd.read_csv("y.csv", header=None)
y = np.array(y)
y = y.flatten()


plt.figure(figsize=(10, 10))
for index, (image, label) in enumerate(zip(X[3497:3502], y[3497:3502])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (20, 20)), cmap=plt.cm.gray)
    plt.title(' %i\n' % label, fontsize=20)


theta = np.zeros(len(X[0]))
ynew = np.zeros(len(y))
data = []

# Calculating the error %
for j in range(10):
    for i in range(l):
        if(y[i] == j):
            ynew[i] = 1
        else:
            ynew[i] = 0
    hypothesis = predict(X, ynew, theta, 5000, 1)
    print("hypothesis for {} ".format(j), hypothesis)
    hypothesis = hypothesis.tolist()
    data.append(hypothesis)
data = np.asarray(data)
error = np.zeros(len(y))
Err = [error for j in range(10)]
for j in range(10):
    for i in range(l):
        if(y[i] == j):
            error[i] = 100*(1 - data[j][i])
        else:
            error[i] = 100*data[j][i]
    Err[j] = error
    print("error percentages for {} ".format(j), error)

# Generating the confusion matrix
confusion = np.zeros((10, 10))
for j in range(10):
    for i in range(l):
        if(500*j <= i and i < 500*(i+1) and data[j][i] > 0.5):
            confusion[j][math.floor(
                i/500)] = confusion[j][math.floor(i/500)] + 1

        elif ((500*j > i or i >= 500*(i+1)) and data[j][i] > 0.5):
            confusion[j][math.floor(
                i/500)] = confusion[j][math.floor(i/500)] + 1
print(confusion)
