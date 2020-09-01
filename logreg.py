'''
Logistic Regression
This class implements linear regressoin for machine learning with 1 layer.
'''

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    '''
    Logistic regression 
    '''
    
    def fit(self, X, Y):
        '''
        Fit the logistic regression model size to the data.
        X ... data matrix, rows are features, columns are instances, size (n_x, m)
        Y ... classification of instances, size (1, m)
        '''
        self.n_x = X.shape[0]
        self.m = X.shape[1]
        self.X = X
        self.Y = Y
        self.w = np.zeros((X.shape[0], 1))
        self.b = 0.0
        self.costs = []
        self.learning_iter = []
    
    def train(self, n_iterations, learning_rate=0.01, verbose=False):
        '''
        Train the network and determine the values of b and w.
        n_ierations ... number of optimisation cycles
        learning_rate ... gradient descent factor
        verbose ... print the cost function value
        '''
        
        costs = []
        learning_iter = []
        w = self.w
        b = self.b 
        
        for i in range(n_iterations):
            
            # calculate derivatives
            grads, cost = self._propagate(w, b)
            dw = grads['dw']
            db = grads['db']
            
            # Update new parameters
            w = w - learning_rate * dw
            b = b - learning_rate * db
            
            # Store cost value
            if i % 100 == 0:
                costs.append(cost)
                learning_iter.append(i)
            
            # Print info
            if i % 100 == 0:
                print('Cost at iteration {} is {}.'.format(i, cost))
        
        self.w = w
        self.b = b
        self.costs = costs
        self.learning_iter = learning_iter
    
    def predict(self, x_test):
        '''
        Predict classes based on learnt parameters w and b.
        x_test ... test values of dimension features X instances
        '''
        predictions = self.sigmoid( np.dot(np.transpose(self.w), x_test) + self.b )
        predictions = predictions > 0.5
        
        return predictions.astype(int)
    
    def print_accuracy(self, predictions, true_values):
        '''
        Print training and test accuracy.
        '''
        train_accuracy = 1 - np.mean(np.abs( self.Y - self.predict(self.X) ))
        print('Training accuracy: {} %.'.format(100*train_accuracy))
        
        test_accuracy = 1 - np.mean(np.abs( true_values - predictions ))
        print('Test accuracy: {} %.'.format(100*test_accuracy))
    
    def plot_cost(self):
        '''
        Plot training loss.
        '''
        plt.plot(self.learning_iter, self.costs)
        plt.xlabel('Learning Iteration')
        plt.ylabel('Cost')
        plt.title('Cost Value Learning')
        plt.show()
    
    def _propagate(self, w, b):
        '''
        Calculate forward and backward propagaiton for the optimisaiton step.
        '''
        A = self.sigmoid(np.dot( np.transpose(w), self.X ) + b)
        cost = self._cost_function(A)
        
        # Compute derivatives for backpropagation (dw := dcost/dw, db := dcost/db)
        dw = 1/self.m * np.dot(self.X, np.transpose(A - self.Y))
        db = 1/self.m * np.sum( A - self.Y )
        grads = {'dw':dw, 'db':db}
        
        return grads, cost
    
    def _cost_function(self, A):
        '''
        Calculate logistic regression cost function.
        '''
        cost = -1/self.m * np.sum( self.Y * np.log(A) + (1 - self.Y) * np.log(1 - A) )
        return np.squeeze(cost)
    
    def reset_parameters(self):
        '''
        Reset parameters w and b to zero for second training.
        '''
        self.w = np.zeros((dim, 1))
        self.b = 0
        self.costs = []
        self.learning_iter = []
    
    @staticmethod
    def sigmoid(x):
        '''
        Calulates sigmoid function for a number x, vector or a matrix.
        '''
        s = 1 / (1 + np.exp(-x))
        return s
    
    @staticmethod
    def sigmoid_derivative(x):
        '''
        Cmpute first derivative of sigmois as ds/dx.
        '''
        s = sigmoid(x)
        ds = s * (1 - s)
        return ds
