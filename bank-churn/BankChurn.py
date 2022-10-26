#!/usr/bin/env python
# coding: utf-8

# # Predicting Bank Customer Churn
# This is an exampel for solving classification problems with Python’s Scikit-learn library
# 
# ## Installing Required Libraries
# - pip3 install numpy
# - pip3 install pandas
# - pip3 install matplotlib
# - pip3 install scikit-learn

# In[64]:


# Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing the Dataset
# The dataset for this problem can be downloaded freely from this [kaggle link](https://www.kaggle.com/datasets/hj5992/bank-churn-modelling).
# 
# Download the dataset locally and then execute the following command to load the data. 

# In[79]:


dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head()


# In[80]:


dataset.dtypes


# The output shows that the dataset has 14 columns. 
# 
# Based on the first 13 columns, we have to predict the value in the exited column i.e. whether or not the customer will exit the bank within 6 months after the data for the first 13 columns is recorded.

# # Preprocessing
# Some of the columns in our dataset are totally random and do not help us indicate whether or not a customer will leave the bank.
# 
# - RowNumber
# - CustomerId
# - Surname 
# 
# columns do not play any part in a customer’s decision to churn or stay with a bank. 
# 
# Remove these three columns from the dataset.

# In[66]:


dataset = dataset.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
dataset.head()


# # Scikit-learn
# The Scikit Learn library work with numerical features. Some of the features in our dataset are categorical, like *Geography* and *Gender* which contain textual data.
# 
# To convert textual features into numeric features, we can use the one-hot encoding technique. In one-hot encoding, a new column is created for each unique value in the original column.
# 
# The integer 1 is added to one of the new columns that corresponds to the original value. 
# In the remaining columns 0s are added.
# 
# Let’s first remove the categorical columns **Geography** and **Gender** from the dataset and create a temp_dataset without these columns.

# In[67]:


temp_dataset =  dataset.drop(['Geography', 'Gender'], axis=1)
temp_dataset.head()


# Use Pandas **get_dummies()** method to convert the categorical columns into numeric columns.

# In[68]:


onehot_geo = pd.get_dummies(dataset.Geography)
onehot_gen = pd.get_dummies(dataset.Gender)
onehot_geo.head()


# The output shows that 1 has been added in the column for France in the first row, while the Germany and Spain columns contain 0. 
# 
# This is because in the original Geography column, the first row contained France.
# 
# Take these new columns and add them to our temp_dataset where we dropped the original Geography and Gender columns.

# In[69]:


final_dataset = pd.concat([temp_dataset, onehot_geo, onehot_gen], axis=1)
final_dataset.head()


# # Divide the Data into Training and Test
# You need to divide our dataset into training and test sets. 
# 
# The training set will be used to train the machine learning classifiers, while the test set will be used to evaluate the performance of our classifier.
# 
# Before dividing the data into training and test set, we need to divide the data into features and labels. 
# 
# The feature set contains independent variables, and the label set contains dependent variable or the labels that you want to predict. 
# 
# The following script divides the dataset into feature set X and label set y.

# In[70]:


X = final_dataset.drop(['Exited'], axis=1).values
y = final_dataset['Exited'].values


# You can further divide the data into training and test sets:

# In[71]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Training and Evaluating
# You are now ready to train the machine learning models on the dataset. 
# 
# There are several machine learning classifiers available in the Scikit-learn library. 
# 
# You will be using the Random Forest classifier which is one of the most powerful machine learning classifiers.
# 
# We can train our algorithm using the RandomForestClassifier class from the sklearn.ensemble module. 
# 
# To train the algorithm we need to pass the training features and labels to the fit() method of the RandomForestClassifier class. 
# 
# To make predictions on the test set, the predict() method is used as shown below:

# In[72]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=200, random_state=0)  
clf.fit(X_train, y_train)  
y_pred = clf.predict(X_test)


# Once the model is trained, you can use a variety of evaluation metrics to measure the algorithm’s performance on your test set. 
# 
# Some of the most commonly used measures are accuracy, confusion matrix, precision, recall and F1. 
# 
# The following script calculates these values for the model when the model is evaluated on your test set.

# In[73]:


from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_test, y_pred))  
print(accuracy_score(y_test, y_pred ))


# The output shows that our model achieves an accuracy of 86% on the test set which is pretty good.

# # Test dataset
# - 605,54,2,56897,1,1,1,250765,0,0,1,0,1
# - 100,54,2,1000,1,1,1,100500,0,0,1,1,0

# In[77]:


print(clf.predict([ [605,54,2,56897,1,1,1,250765,0,0,1,0,1] ]))


# In[78]:


print(clf.predict([ [100,54,2,1000,1,1,1,100500,0,0,1,1,0] ]))


# In[ ]:




