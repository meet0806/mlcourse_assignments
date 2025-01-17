import numpy as np


class LinearRegression:
    
    def __init__(self, lr=0.001, n_iters=1000):

        #lr is learning rate
        self.lr = lr
        
        #n_iters is number of iterations
        self.n_iters = n_iters
        
        #weights and bias are slope and intercept of the line
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        
        n_samples, n_features = X.shape

        #initializing weights and bias with zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        #gradient descent logic
        for _ in range(self.n_iters):

            y_predicted = np.dot(X, self.weights) + self.bias
            #dw is the gradient of the loss function with respect to the weights
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            #db is the gradient of the loss function with respect to the bias
            db = (1 / n_samples) * np.sum(y_predicted - y)

            #updating the weights and bias by subtracting the gradient multiplied by the learning rate
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
    
    
    def predict(self, X):
        
        #predicting the values using y = mx + c equation
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
    
