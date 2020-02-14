import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class Modeler:
    '''
    model the classes of input and output the probability
    '''

    def __init__(self, batch_size=1):
        # train callback multiple
        self.batch_size = batch_size

        # the input data
        self.x = []
        
        # the output data
        self.y = []
        
        #initialize model for classification and prediction
        self.initialize_classifier()

        # the model has not been fitted yet
        self.is_fitted = False
        
    def initialize_classifier(self):
        '''
        initialize model for fitting and prediction
        '''
        # define model
        self.classifier = LogisticRegression(solver='liblinear')
    
    def add_data(self, context, label):
        '''
        appends data x and y to current object data set
        '''

        self.x.append(list(map(ord, context)))
        self.y.append(label)

        if len(self.y) % self.batch_size == 0:
            self.train()
    
    def predict(self, context):
        '''
        predict the probability that the context fits the model input description
        '''
        # check if model is fitted
        if not self.is_fitted:

            # if not then train the model
            if self.train() < 0:
                return -1
        
        # if all is well use model to predict
        return self.classifier.predict(np.array([list(map(ord, context))]))[0]
        
    def train(self, verbose=0):
        # convert your data to numpy array
        x, y = np.array(self.x), np.array(self.y)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        
        # check if the min number of classes is 2
        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            return -1

        # fit the classifier on the dataset
        self.classifier.fit(X_train, y_train)
        
        # set is fitted to true if fitted
        if not self.is_fitted:
            self.is_fitted = True

        # make class predictions with the model
        y_pred = self.classifier.predict(X_test)
        
        # what to display
        if verbose:
            for i in range(len(X_test)):
                score = '*' if y_test[i] != y_pred[i] else ''
                print(f"{X_test[i].tolist().__str__():15s} expected => {y_test[i]}, predicted => {y_pred[i]} {score}")
        
        # calc accuracy
        accuracy = self.classifier.score(X_test, y_test)
        
        # return the accuracy
        return accuracy
