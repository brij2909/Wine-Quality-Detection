# Import The Dependencies
# Import Libraries

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sbn
import sys
import random as rnd 

# Machine Learning Libraries

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Data Collection
# Loading the dataset to a Pandas DataFrame

df = pd.read_csv(r"C:\Users\Dell\Desktop\AI\Machine Learning\Wine Quality\WINEMODEL\winequality-red.csv")
print(df.shape)
print('---------------------------------------------------------------')
print(df.head())
print('---------------------------------------------------------------')
# Checking the missing values is dataset
print(df.isnull().sum())
print('---------------------------------------------------------------')

# Data Analysis and Visualization
# Statistical Measures of the dataset
print(df.describe())
print('---------------------------------------------------------------')
# Number of values of each quality
sbn.catplot(x = 'quality', data=df, kind='count')
plt.show()
print('---------------------------------------------------------------')
# Volatile acidity vs. quality
plot = plt.figure(figsize=(5, 5))
sbn.barplot(x = 'quality', y = 'volatile acidity', data=df)
plt.show()
print('---------------------------------------------------------------')
# citric acid vs. quality
plot = plt.figure(figsize=(5, 5))
sbn.barplot(x='quality', y='citric acid', data=df)
plt.show()
print('---------------------------------------------------------------')

# Correlation of columns
# 1. Positive Correlation
# 2. Negative Correlation
correlation = df.corr()
# Constructing the heatmap to understand the correalation between the columns
plt.figure(figsize=(10, 10))
sbn.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
plt.show()
print('---------------------------------------------------------------')

# Data Preprocessing
# Seoarate the data and label
X = df.drop('quality', axis=1)
print(X)

# Label Binarization
Y = df['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
print(Y)
print('---------------------------------------------------------------')
print(df['quality'].head(15))
print('---------------------------------------------------------------')

# Train & Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(Y.shape, Y_train.shape, Y_test.shape)
print('---------------------------------------------------------------')

# Model Training and Model Evaluation
# RandomForestClassifier & Accuracy Score

model = RandomForestClassifier()
model.fit(X_train, Y_train)
# model accuracy on test data

X_test_predict = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predict, Y_test)*100
print("Accuracy is ", test_data_accuracy)

import pickle

# Assuming 'model' is your trained ML model
wine_model = pickle.dumps(model)

# save the model as a pickle file
with open('model.pkl', 'wb') as file:
    file.write(wine_model)

# Load the model from the pickle file
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.loads(file.read())

# Assuming 'X_test' is your test data
predictions = loaded_model.predict(X_test)

# Joblib

from joblib import dump, load

# assuming 'model' is your trained ML model
dump(model, 'model.joblib')

# load the model from the joblib file
loaded_model = load('model.joblib')

# assuming 'X_test' is your test data
predictions = loaded_model.predict(X_test)