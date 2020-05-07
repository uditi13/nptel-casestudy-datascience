# -*- coding: utf-8 -*-
"""
Created on Thu May  7 01:15:06 2020

@author: Uditi
"""


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Setting dimensions for all plots
sns.set(rc={'figure.figsize':(9,4)})

# Importing data from csv
carsdata=pd.read_csv('C:\\Users\\Uditi\\Desktop\\cars_sampled.csv')

cars1=carsdata.copy(deep=True)

# Gives info about data type of each column
cars1.info()

# Summary of data
cars1.describe()

# Changes the number of digits after decimal point of 
# every float data in dataframes read by pandas to 3 digits
pd.set_option('display.float_format',lambda x:'%.3f' % x)
cars1.describe()
        
# On using describe(), we can't see all columns sometimes
# To tackle this problem:
pd.set_option('display.max_columns',500)
cars1.describe()

#Dropping unwanted columns
cols=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars1=cars1.drop(columns=cols,axis=1)

# Deleting duplicate records from Dataframe
cars1.drop_duplicates(keep='first',inplace=True)

#Calculating number of missing values
cars1.isnull().sum()

# The data is very dispersed and thus, we need to set working ranges.

# To check the frequency of each value in a column
yearwise_count=cars1['yearOfRegistration'].value_counts()

#A scatter plot or a box plot will show how concentrated the data is in some places
#and how outliers are affecting the data.

sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=cars1) 

sum(cars1['yearOfRegistration']>2018)   #only 26
sum(cars1['yearOfRegistration']<1950)   #only 38
#These values were selected by hit and trial.
#Hence, we can sustain this loss of data and can keep the range from 1950-2018

#We do the same analysis for the variable price.
#Working range for price comes out to be: 100-150000 

#We do the same analysis for the variable powerPS.
#Working range for powerPS comes out to be: 10-500 

#Implementing the working ranges.
cars1=cars1[
            (cars1.yearOfRegistration<=2018)
        &   (cars1.yearOfRegistration>=1950)
        &   (cars1.price<=150000)
        &   (cars1.price>=100)
        &   (cars1.powerPS<=500)
        &   (cars1.powerPS>=10)
        
        ]

#If we have year of registration and months, we can substitute with a variable age.
# Take the difference between current year and the registration years, then divide month by 12
# and add to the difference.
cars1['monthOfRegistration']/=12
cars1['Age']=(2018-cars1['yearOfRegistration'])+cars1['monthOfRegistration']
#Rounding off to 2 decimals
cars1['Age']=round(cars1['Age'],2)

#Dropping year and month
cars1=cars1.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

#You can visualise new parameters now.
sns.distplot(cars1['Age'])

#Always check boxplot of Price vs any variable to check the dependency 
#of price on that variable
# Now we have checkted the significance of numerical variables.
# Checking for categorical.

#Same process for every categorical variable
cars1['seller'].value_counts()
sns.countplot(x='seller',data=cars1)

cars1['abtest'].value_counts()
sns.countplot(x='abtest',data=cars1)

# If in any case, one value overpowers others, like in the case of the seller variable
# That variable is insignificant


# If in any case, two values are equally distributed, 
# like in the case of the abtest variable
# That variable is insignificant

# Always check boxplot of variables vs price. To see if there is difference in output 
# for each type of value in the variable. 

sns.boxplot(y='price',x='vehicleType',data=cars1)

# We find after analysis of all variables that,
# seller,offerType,abtest are insignificant

cars1=cars1.drop(columns=['seller','offerType','abtest'],axis=1)

#========================================================
# CORRELATION
#===============================================================================
# Check correlation between numerical values to check dependencies.
cars1.corr()
cars_select1=cars1.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]
#To check dependency on Price we did this

#===========================================================

"""
We are going to build a Linear Regression and Random Forest Model
on two sets of data.
1. Data obtained by omitting rows with any missing values
2. Data obtained by imputing the missing values

"""

# Two approaches: 1) Omitting rows
#                 2) Imputing values

#====================================================================
# OMITTING MISSING VALUES
#===========================================================


# Omitting rows:

cars1_omit=cars1.dropna(axis=0) 

#Converting categorical variables to dummy variables
cars1_omit=pd.get_dummies(cars1_omit,drop_first=True)



#=========================================================
# IMPORTING NECESSARY LIBRARIES
#========================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#===========================================================
# MODEL BUILDING WITH OMITTED DATA
#==================================================================

# Separating dependent and independent variables
x1=cars1_omit.drop(['price'],axis='columns',inplace=False)
y1=cars1_omit['price']

#Plotting the variable price
prices = pd.DataFrame({"1. Before":y1, "2. After":np.log(y1)})
prices.hist()
#We see that the before part is skewed and after is bell shaped or normal


# TRANSFORMING PRICE AS LOG VALUE
# Here we use the strategy of taking the natural log of the price column,
# This is done because the data is very skewed in its natural form.
# After taking the natural log, we can understand it better.

#Therefore,
y1=np.log(y1) 

# SPLITTING DATA INTO TEST AND TRAIN

train_x,test_x,train_y,test_y=train_test_split(x1,y1,test_size=0.3,random_state=3)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)


#====================================================
# BASELINE MODEL FOR OMITTED DATA
#=====================================================
# We are making a base model by using test data mean value

# PERFORMANCE METRICS/ BASELINE MODEL
#finding the mean for test data value
base_pred=np.mean(test_y)
print(base_pred)

# Repeating same value till the length of test data to ensure same number of rows for
# Matrix comparison.
base_pred=np.repeat(base_pred,len(test_y))


base_root_mean_square=np.sqrt(mean_squared_error(test_y,base_pred))
print(base_root_mean_square)
# The root mean squared error for predictions should be less than the rms error value
# for base predictions

#=============================================================
# LINEAR REGRESSION MODEL
#==========================================================

#Setting intercept as true
linear=LinearRegression(fit_intercept=True)

#MODEL
linear.fit(train_x,train_y)

#Predicting model on test set
predictions=linear.predict(test_x)

#Computing RMSE for prediction
rmse=np.sqrt(mean_squared_error(test_y,predictions))
print(rmse)
# Since value of rmse of predictions is much less than the rmse of base model,
# Our linear regression model is working very well.



# R squared value
r2_test=linear.score(test_x,test_y)
r2_train=linear.score(train_x,train_y)
print(r2_test,r2_train)
# If the r2 scores for both test and train are similar, model is working well.



#Regression diagnostics - Residual plot analysis

# Residuals are the differences between the predicted data and test data.
# Residual plot has Residuals on the y-axis and the predictions for text_x on x-axis
residuals=test_y-predictions
sns.regplot(x=predictions,y=residuals,scatter=True,fit_reg=False)
residuals.describe()

# Closer the mean is to zero for residuals, the better. 
# Hence the errors should be less.
# Closer the scatter plot is to zero for residuals, the better.




# Do the same after imputing data!
# Results might vary a lot.


#======================================================================
# RANDOM FOREST REGRESSOR MODEL
#========================================================

#Model parameters
rf= RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,
                          min_samples_split=10,min_samples_leaf=4,random_state=1)

# Model
rf.fit(train_x,train_y)

# Predicting model on test set
predictions=rf.predict(test_x)


# Computing RMSE for prediction
rmse=np.sqrt(mean_squared_error(test_y,predictions))
# rmse of random forest is lesser than that of linear reg model hence this is better.
# Since value of rmse of predictions is much less than the rmse of base model,
# Our linear regression model is working very well.

# R squared value
r2_test=rf.score(test_x,test_y)
r2_train=rf.score(train_x,train_y)
print(r2_test,r2_train)
# If the r2 scores for both test and train are similar, model is working well.

# Do the same after imputing data!
# Results might vary a lot.


#==============================================================
# MODEL BUILDING WITH IMPUTED DATA
#==================================================
# Replacing missing values w mean/median (only numeical)
cars_imputed = cars1.apply(lambda x:x.fillna(x.median()) \   
                          if x.dtype=='float' else \
                          x.fillna(x.value_counts().index[0])) 
# Here the lambda function will fill the missing values with MEDIAN if the dtype is float else
# else it will fill the missing values with most frequent value/ the MODE
# First applicable for Float and Second for Object
    
cars_imputed.isnull().sum()

#Converting categorical variables to dummy variables
cars_imputed=pd.get_dummies(cars_imputed, drop_first=True)

#==============================================================================================
# MODEL BUILDING WITH IMPUTED DATA
#=======================================================


# Separating input and output features
x2 = cars_imputed.drop(['price'], axis='columns', inplace=False)
y2 = cars_imputed['price']

#Plotting the variable price
prices = pd.DataFrame({"1. Before":y2, "2. After":np.log(y2)})
prices.hist()
#We see that the before part is skewed and after is bell shaped or normal


# TRANSFORMING PRICE AS LOG VALUE
# Here we use the strategy of taking the natural log of the price column,
# This is done because the data is very skewed in its natural form.
# After taking the natural log, we can understand it better.

#Therefore,
y2=np.log(y2) 

# SPLITTING DATA INTO TEST AND TRAIN

train1_x,test1_x,train1_y,test1_y=train_test_split(x2,y2,test_size=0.3,random_state=3)
print(train1_x.shape, test1_x.shape, train1_y.shape, test1_y.shape)


#=========================================================
# BASELINE MODEL FOR IMPUTED DATA
#==============================================================

# We are making a base model by using test data mean value

# PERFORMANCE METRICS/ BASELINE MODEL
#finding the mean for test data value
base_pred=np.mean(test1_y)
print(base_pred)

# Repeating same value till the length of test data to ensure same number of rows for
# Matrix comparison.
base_pred=np.repeat(base_pred,len(test1_y))


base_root_mean_square=np.sqrt(mean_squared_error(test1_y,base_pred))
print(base_root_mean_square)
# The root mean squared error for predictions should be less than the rms error value
# for base predictions

#==========================================================
# LINEAR REGRESSION WITH IMPUTED DATA
#==========================================================

#Setting intercept as true
linear2=LinearRegression(fit_intercept=True)

#MODEL
linear2.fit(train1_x,train1_y)

#Predicting model on test set
predictions2=linear2.predict(test1_x)

#Computing RMSE for prediction
rmse2=np.sqrt(mean_squared_error(test1_y,predictions2))
print(rmse2)
# Since value of rmse of predictions is much less than the rmse of base model,
# Our linear regression model is working very well.



# R squared value
r2_test2=linear2.score(test1_x,test1_y)
r2_train2=linear2.score(train1_x,train1_y)
print(r2_test2,r2_train2)
# If the r2 scores for both test and train are similar, model is working well.



#Regression diagnostics - Residual plot analysis

# Residuals are the differences between the predicted data and test data.
# Residual plot has Residuals on the y-axis and the predictions for text_x on x-axis
residuals2=test1_y-predictions2
sns.regplot(x=predictions2,y=residuals2,scatter=True,fit_reg=False)
residuals2.describe()

# Closer the mean is to zero for residuals, the better. 
# Hence the errors should be less.
# Closer the scatter plot is to zero for residuals, the better.


#======================================================================
# RANDOM FOREST REGRESSOR MODEL
#========================================================

#Model parameters
rf2= RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,
                          min_samples_split=10,min_samples_leaf=4,random_state=1)

# Model
rf2.fit(train1_x,train1_y)

# Predicting model on test set
predictions_rf2=rf2.predict(test1_x)


# Computing RMSE for prediction

rmse_rf2=np.sqrt(mean_squared_error(test1_y,predictions_rf2))
# rmse of random forest is lesser than that of linear reg model hence this is better.
# Since value of rmse of predictions is much less than the rmse of base model,
# Our linear regression model is working very well.

# R squared value
r2_test_rf2=rf2.score(test1_x,test1_y)
r2_train_rf2=rf2.score(train1_x,train1_y)
print(r2_test_rf2,r2_train_rf2)
# If the r2 scores for both test and train are similar, model is working well.

# PRINT ALL VALUES FOR OMITTED AND IMPUTED DATA!!
# R SQUARED VALUE FOR TRAIN,TEST OF LINEAR REG AND RANDOM FOREST EACH FOR BOTH OMITTED AND IMPUTED DATA
# BASE RMSE VALUES FOR OMITTED AS WELL AS IMPUTED
# RMSE VALUE FOR TEST FROM LINEAR REG AND RANDOM FOREST FOR OMITTED AND IMPUTED DATA
  

# WE FIND THAT RANDOM FOREST PERFORMS BETTER IN BOTH CASES
# AS THE RMSE VALUE IS A LOT LESSER THAN THAT OF LINEAR REG



#========================================================
# END 
#============================================================






