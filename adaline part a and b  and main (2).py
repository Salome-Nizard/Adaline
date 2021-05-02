#!/usr/bin/env python
# coding: utf-8

# In[1]:



class Adaline(object):
    """ Adaline (Adaptive Linear Neuron) for binary classification.
        Minimises the cost function using gradient descent. """

    def __init__(self, learn_rate = 0.01, iterations = 100):
        self.learn_rate = learn_rate
        self.iterations = iterations


    def fit(self, X, y, biased_X = False, standardised_X = False):
        """ Fit training data to our model """
        if not standardised_X:
            X = self._standardise_features(X)
        if not biased_X:
            X = self._add_bias(X)
        self._initialise_weights(X)
        self.cost = []

        for cycle in range(self.iterations):
            output_pred = self._activation(self._net_input(X))
            errors = y - output_pred   
            self.weights += (self.learn_rate * X.T.dot(errors))
            cost = (errors**2).sum() / 2.0
            self.cost.append(cost)
        return self
    
    def score(self, X, y):
        """
        Model score is calculated based on comparison of
        expected value and predicted value.
        :param X:
        :param y:
        :return:
        """
        wrong_prediction = abs((self.predict(X) - y) / 2).sum()
        self.score_ = (len(X) - wrong_prediction) / len(X)
        return self.score_


    def _net_input(self, X):
        """ Net input function (weighted sum) """
        return np.dot(X, self.weights)


    def predict(self, X, biased_X=False):
        """ Make predictions for the given data, X, using unit step function """
        if not biased_X:
            X = self._add_bias(X)
        return np.where(self._activation(self._net_input(X)) >= 0.0, 1, 0)


    def _add_bias(self, X):
        """ Add a bias column of 1's to our data, X """
        bias = np.ones((X.shape[0], 1))
        biased_X = np.hstack((bias, X))
        return biased_X


    def _initialise_weights(self, X):
        """ Initialise weigths - normal distribution sample with standard dev 0.01 """
        random_gen = np.random.RandomState(1)
        self.weights = random_gen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        return self
    
    
    def _standardise_features(self, X):
        """ Standardise our input features with zero mean and standard dev of 1 """
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis = 0)
        return X_norm


    def _activation(self, X):
        """ Linear activation function - simply returns X """
        return X

      


# In[7]:


import numpy as np
from numpy import random as rand
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

import seaborn as sns

def plotDecReg(X, y, classifier):
    # plot the decision surface
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1
    X_1, X_2 = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
    pred = classifier.predict(np.array([X_1.flatten(), X_2.flatten()]).T)
    pred = pred.reshape(X_1.shape)
    colors = ListedColormap(('red', 'blue'))

    # showed  prediction
    plt.contourf(X_1, X_2, pred, alpha=0.1, cmap=colors)
    plt.xlim(X_1.min(), X_1.max())
    plt.ylim(X_2.min(), X_2.max())

    plt.scatter(x=X[y == -1, 1], y=X[y == -1, 0],
                alpha=0.9, c='red',
                marker='s', label=-1.0)

    plt.scatter(x=X[y == 1, 1], y=X[y == 1, 0],
                alpha=0.9, c='blue',
                marker='x', label=1.0)







def partA():
    print("Start Part A\n")

    # create a Adaline classifier and train on our data
    dataSize = 1000
#    X, y = creatData(dataSize, "A", 100)
    #initData
    rand.seed(10)
    data = rand.uniform(-1, 1, size=(dataSize, 2))  # any value

    train = np.zeros(dataSize)
    for i in range(dataSize):
        if data[i][0] > 0.5 and data[i][1] > 0.5:
            train[i] = 1
        else:
            train[i] = -1
    x = data.astype(np.float64)  # Copy of test array & cast
    y = train.astype(np.float64) # Copy of train array & cast    
    

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
# create Adaline & train on Data
    #rate= 0.01 , data= 1,000 , n= 100
    classifierAda = Adaline(0.00001, 100).fit(x, y)
    ax[0].plot(range(1, len(classifierAda.cost) + 1), classifierAda.cost, marker='o')
    ax[0].set_xlabel('Iter')
    ax[0].set_ylabel('Cost')
    ax[0].set_title('Part A:Adaline: data 1,000 ,rate: 0.01 ,n = 100')

    #rate: 0.00001 , data = 1,000 , n = 100
    print("rate= 1/100 , data= 1000 , n= 100\n")
    print("The Score: ", classifierAda.score(x, y) * 100,"%\n")
    print("Cost: ", np.array(classifierAda.cost).min() , "\n")

    # plot error after each training iter
    plotDecReg(x, y, classifier=classifierAda)
    plt.title('Part A: Adaline: data 1,000 , rate: 0.00001 , n = 100')
    plt.legend(loc='upper left')
    plt.show()

    #confusion_matrix
    cmat = confusion_matrix(classifierAda.predict(x), y)
    plt.subplots()
    sns.heatmap(cmat, fmt=".0f", annot=True)
    plt.title("confusion matrix: data 1,000 , rate: 0.00001 , n = 100")
    plt.xlabel("Test")
    plt.ylabel("Predict")
    plt.show()

    # rate= 0.0001 , data= 1,000 , n= 100
    classifierAda2 = Adaline(0.0001,50).fit(x, y)
    ax[1].plot(range(1, len(classifierAda2.cost) + 1), classifierAda2.cost, marker='o')
    ax[1].set_xlabel('Iter')
    ax[1].set_ylabel('Cost')
    ax[1].set_title('Part A:Adaline: data 1,000 , rate: 0.0001 , n = 100')
    plt.show()

    #rate: 0.0001 , data = 1,000 , n = 100
    print("rate: 1/10,000 , data = 1,000 , n = 100\n")
    print("The Score: ", classifierAda2.score(x, y) * 100, "%\n")
    print("Cost: ", np.array(classifierAda2.cost).min())

    # plot error after each training iter 
    plotDecReg(x, y, classifier=classifierAda2)
    plt.title('Part A: Adaline: data 1,000 , rate: 0.0001 , n = 100')
    plt.legend(loc='upper left')
    plt.show()

    # confusion_matrix
    cmat2 = confusion_matrix(classifierAda2.predict(x), y)
    plt.subplots()
    sns.heatmap(cmat2, fmt=".0f", annot=True)
    plt.title("confusion matrix , data 1,000 , rate: 0.0001 , n = 100")
    plt.xlabel("Test")
    plt.ylabel("Predict")
    plt.show()

    dataSize = 10000
    # initData
    rand.seed(10)
    data = rand.uniform(-1, 1, size=(dataSize, 2))  # any value

    train = np.zeros(dataSize)
    for i in range(dataSize):
        if data[i][0] > 0.5 and data[i][1] > 0.5:
            train[i] = 1
        else:
            train[i] = -1
    x_3 = data.astype(np.float64)  # Copy of test array & cast
    y_3= train.astype(np.float64)  # Copy of train array & cast


    classifierAda3 = Adaline(0.001, 75).fit(x_3, y_3)

    #rate: 0.001 , data = 1,000 , n = 10,000
    print("rate: 1/100 , data = 1,000 , n = 10,000\n")
    print("The Score: ", classifierAda3.score(x_3, y_3) * 100, "%")
    print("Cost: ", np.array(classifierAda3.cost).min())

    # plot error after each training iter
    plotDecReg(x_3, y_3, classifier=classifierAda3)
    plt.title('Part A: Adaline: data 10,000 , Learning rate: 0.001, n = 10,000')
    plt.legend(loc='upper left')
    plt.show()

    #confusion_matrix
    cmat3 = confusion_matrix(classifierAda3.predict(x_3), y_3)
    plt.subplots()
    sns.heatmap(cmat3, fmt=".0f", annot=True)
    plt.title("confusion matrix: data 10,000 , rate:0.001 , n = 10,000")
    plt.xlabel("Test")
    plt.ylabel("Predict")
    plt.show()

def partB():
    print("\nPart B\n")
    dataSize = 1000

    # data = 1,000
    rand.seed(10)
    data = rand.uniform(-1, 1, size=(dataSize, 2))  # any value
    train = np.zeros(dataSize)

    for i in range(dataSize):
        if 0.5 <= (data[i][0] * 2 + data[i][1] * 2) <= 0.75:
            train[i] = 1
        else:
            train[i] = -1

    X_1 = data.astype(np.float64)  # Copy of test array & cast
    y_1 = train.astype(np.float64)  # Copy of train array & cast
    
    classifierAda = Adaline(0.00001, 100).fit(X_1, y_1)
    # data = 100,000



    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # learning rate = 0.0001 || d_type = 1,000 || n = 100
    ax[0].plot(range(1, len(classifierAda.cost) + 1), classifierAda.cost, marker='o')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Cost')
    ax[0].set_title('Part B: Adaline Algorithm \ndata 1,000 || Learning rate: 0.0001')

    print("Learning rate: 1/10,000 || data = 1,000\n")
    print("score: ", classifierAda.score(X_1, y_1) * 100, "%")
    print("cost: ", np.array(classifierAda.cost).min())

    # plot our miss-classification error after each iteration of training
    plotDecReg(X_1, y_1, classifier=classifierAda)
    plt.title('Part B: Adaline Algorithm \ndata 1,000 || Learning rate: 0.0001')
    plt.legend(loc='upper left')
    plt.show()

    # confusion_matrix
    cm = confusion_matrix(classifierAda.predict(X_1), y_1)
    plt.subplots()
    sns.heatmap(cm, fmt=".0f", annot=True)
    plt.title("confusion matrix \ndata 1,000 || Learning rate: 0.0001")
    plt.xlabel("Test")
    plt.ylabel("Predict")
    plt.show()

    
    
    dataSize = 100000
     # initData
    rand.seed(10)
    data = rand.uniform(-1, 1, size=(dataSize, 2))  # any value

    train = np.zeros(dataSize)
    for i in range(dataSize):
        if 0.5 <= (data[i][0] * 2 + data[i][1] * 2) <= 0.75:
            train[i] = 1
        else:
            train[i] = -1

    X_2 = data.astype(np.float64)  # Copy of test array & cast
    y_2 = train.astype(np.float64)  # Copy of train array & cast.


    classifierAda2 = Adaline(0.00000001, 50).fit(X_2, y_2)
    
    #rate = 0.0001 , d_type = 100,000 , n = 100
    ax[1].plot(range(1, len(classifierAda2.cost) + 1), classifierAda2.cost, marker='o')
    ax[1].set_xlabel('Iter')
    ax[1].set_ylabel('Cost')
    ax[1].set_title('Part B: Adaline: data 100,000 , Learning rate: 0.001')
    plt.show()


    # learning rate = 0.0001 || d_type = 100,000 || n = 100
    print("\nLearning rate: 1/10,000 || data = 100,000\n")
    print("score: ", classifierAda2.score(X_2, y_2) * 100, "%")
    print("cost: ", np.array(classifierAda2.cost).min())

    # plot our miss-classification error after each iteration of training
    plotDecReg(X_2, y_2, classifier=classifierAda2)
    plt.title('Part B: Adaline Algorithm \ndata 100,000 || Learning rate: 0.0001')
    plt.legend(loc='upper left')
    plt.show()

    # confusion_matrix
    cm = confusion_matrix(classifierAda.predict(X_2), y_2)
    plt.subplots()
    sns.heatmap(cm, fmt=".0f", annot=True)
    plt.title("confusion matrix \ndata 100,000 || Learning rate: 0.0001")
    plt.xlabel("Test")
    plt.ylabel("Predict")
    plt.show()


if __name__ == '__main__':
    partA()
    partB()


# In[ ]:




