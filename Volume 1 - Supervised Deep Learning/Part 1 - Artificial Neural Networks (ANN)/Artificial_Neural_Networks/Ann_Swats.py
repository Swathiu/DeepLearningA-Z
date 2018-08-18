
#Data Preprocessing - Part 1

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding Categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_Country = LabelEncoder()
X[:, 1] = labelencoder_X_Country.fit_transform(X[:, 1])

labelencoder_X_Gender = LabelEncoder()
X[:, 2] = labelencoder_X_Gender.fit_transform(X[:, 2])

#To avoid relation between the categorical variables, we do below step
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#Splitting the dataset into the Training Set and Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling - Have to apply compusorily for ANN and deep learning - 
#A lot of computations involved, parallel computations exists. So to ease the computations, we 
#need to apply feature scalin. Also, to avoid one independent variable dominating the other one.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building ANN - Part 2
#Importing keras libraries and packages
#Deep learning network is built based on tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Initializing the ANN with sequence of layers
classifier = Sequential()
#Dense function takes care of randomly initializing weights to small numbers close to 0.
#Number of independent variables in the matrix of features = Number of neurons in the input layer = 11 input nodes
#Forward Propagation - 
# 1. The neuron applies the activation function to the sum. Closer the value of Activation function is to 1, more activated is the neuron.
# 2. Higher the value of activation function is for a neuron, the more impact the neuron is going to have on the network and more the neuron passes on the signal.
#Rectifier function - Best activation function for hidden layer
#Sigmoid Function - Best activation function for output layer

#If the model has overfitting problem, then you should apply dropout to all the hidden layers

#Add input layer and first hidden layer with dropout
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu', input_dim = 11))
#Rate - Fraction/Percentage of neurons that you need to drop/disable at each iteration
classifier.add(Dropout(rate=0.1))

#Adding second hidden layer with dropout, so input_dim not required
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.1))

#Adding output layer 
#In case the dependent variable has more than 2 categories, the output Activation function should be softmax function
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid'))

#Compiling ANN - Applying stochastic gradient method to ANN
#Loss function that runs in Adam Optimizer method
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100 )

# Making predictions and evaluating the model - Part 3

#Predicting the Test Set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (1932 + 177)/2500

new_pred = classifier.predict(sc.fit_transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)

# Evaluating, Improving and Tuning ANN

#Evaluating ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential
from keras.layers import Dense
#Function to build architecture of ANN
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()

#Improving ANN
#Grid search for hyper parameters tuning with k-cross validation
#Tuning ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense
#Function to build architecture of ANN
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25,32], 'epochs': [100, 500], 'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_