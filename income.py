# -*- coding: utf-8 -*-
"""
Created on Tue May  5 00:11:46 2020

@author: Uditi
"""



import pandas as pd


import numpy as np


import seaborn as sns


from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression


from sklearn.metrics  import accuracy_score,confusion_matrix




#importing data
data_income = pd.read_csv('C:\\Users\\Uditi\\Desktop\\income.csv')

#creating copy of original data
data = data_income.copy()

"""
#Exploratory data analysis:

#1.Getting to know the data
#2.Data Preprocessing (Missing Values)
#3.Cross tables and data visualization
"""

#Getting to know the data

#****To check variables' data type
print(data.info())

#****Check for missing values
data.isnull()


print('Data columns with null values:\n',data.isnull().sum())
#****No missing values found!

#****Summary of numerical variables
summary_num = data.describe()
print(summary_num)

#****Summary of categorical variables
summary_cate = data.describe(include = 'O')
print(summary_cate)

#****Frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()

#****Checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
#****There exists ' ?' instead of nan

"""
Go back and read the data by including "na_values[' ?']" to consider ? as nan
"""
data = pd.read_csv('C:\\Users\\Uditi\\Desktop\\income.csv',na_values=[' ?'])

#Data preproccessing

data.isnull().sum()


#for the rows with atleast 1 missing value
missing = data[data.isnull().any(axis=1)]
# axis=1 --> to consider at least one column value is missing in one row

"""Note-
1)Missing values in JobType = 1809
2)Missing values in Occupation = 1816
3)There are 1809 rows where 2 specific columns i.e. occupation and Jobtype have missing values
4)(1816-1809) = 7 --> You still have occupation unfilled for these 7 rows because the Jobtype is 'Never worked'
"""
#Getting rid of the missing values as we don't know where they came from or the relationship b/w them

data2 = data.dropna(axis=0)


#Relationship b/w independent variables
correlation = data2.corr()
#since the values are closer to 0, there is no correlation

#====================================
#Cross tables and Data visualization
#====================================
#Extracting Column names
data2.columns

#====================================
#Gender proportion Table
#====================================
gender = pd.crosstab(index    = data2['gender'],
                     columns  = 'count',
                     normalize= True)

print(gender)

#====================================
#Gender vs Salary Status:
#====================================
gender_salstat = pd.crosstab(index       = data2['gender'],
                             columns     = data2['SalStat'],
                             margins     = True,
                             normalize   = 'index') #Include Rows and columns total
print(gender_salstat)

#==========================================
#Frequency distribution of 'Salary Status'
#==========================================
SalStat = sns.countplot(data2['SalStat'])

""" 75% of people's salray status is <=50,000
    & 25% is >50,000
"""
#############  HISTOGRAM OF AGE  #############
sns.distplot(data2['age'], bins=10, kde=False)
# People with age 20-45 age are high in frequency

#############  Box Plot - Age vs Salary Status  ############
sns.boxplot('SalStat', 'age', data=data2)
data2.groupby('SalStat')['age'].median()

data2 = data.dropna(axis=0)

#==========================================================
# LOGISTIC REGRESSION
#==========================================================

#Reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

#convert categorical to dummy/indicator
new_data=pd.get_dummies(data2, drop_first=True)

#Storing the column names
columns_list=list(new_data.columns)
print(columns_list)

#Separating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

#Storing the output values in y
y=new_data['SalStat'].values
print(y)

#Storing the values from input features
x = new_data[features].values
print(x)

#Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

#Make an instance of the model
logistic = LogisticRegression()


#Fitting the values for x & y
logistic.fit(train_x, train_y)
logistic.coef_
logistic.intercept_


#Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

#Confusion Matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

#Calculating the accuracy
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

#Printing the misclassified values from prediction

print('Misclassified samples: %d' % (test_y != prediction).sum())

#============================================================
#  LOGISTIC REGRESSION - REMOVING THE INSIGNIFICANT VARIABLES
#=============================================================

#Reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0, ' greater than 50,000':1})
print(data2['SalStat'])

#storing insignificant variables in cols and removing them
cols = ['gender', 'nativecountry','race','JobType']
new_data = data2.drop(cols,axis =1)

new_data=pd.get_dummies(new_data, drop_first=True)

#Storing the column names
columns_list=list(new_data.columns)
print(columns_list)

#Separating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)


#Storing the output values in y
y=new_data['SalStat'].values
print(y)


#Storing the values from input features
x = new_data[features].values
print(x)


#Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

#Make an instance of the model
logistic = LogisticRegression()

#Fitting the values for x & y
logistic.fit(train_x, train_y)
logistic.coef_
logistic.intercept_

#Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

from sklearn.metrics import accuracy_score

#Calculating the accuracy
accuracyscore=accuracy_score(test_y, prediction)
print(accuracyscore)

#Printing the misclassified values from prediction

print('Misclassified samples: %d' % (test_y != prediction).sum())


#==============================================================
# KNN
#==============================================================

#importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier

#import library for plotting
import matplotlib.pyplot as plt


#Storing the K nearest neighbors classifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 5) # K value is 5

# Fitting the values for X and Y
KNN_classifier.fit(train_x, train_y)

# Predicting the test values with model
prediction = KNN_classifier.predict(test_x)


from sklearn.metrics import confusion_matrix
# Performance metric check
confusionmatrix = confusion_matrix(test_y, prediction)
print('\t','Predicted values')
print("Original values", "\n",confusionmatrix)

#Calculating the accuracy
accuracy_s = accuracy_score(test_y, prediction)
print(accuracy_s)


print('Misclassified samples: %d' % (test_y != prediction).sum())

"""
Effect of K value on classifier
"""

Misclassified_sample = []

# Calculating error for K values b/w 1 and 20

for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())
    
print(Misclassified_sample)
#=========================================
#   END OF SCRIPT
#=========================================





























