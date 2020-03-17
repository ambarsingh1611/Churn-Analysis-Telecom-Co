# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 21:04:41 2020

@author: Cosmic Dust
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.getcwd()
data = pd.read_csv('Telco-Customer-Churn.csv')

data.info()

# Convert data type of Total Charges Column 
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors ='coerce')

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.25, random_state = 101)

train.shape
test.shape

# Which customer has highest probability to switch to another telecom?
# What could be the plausible reason that th churn is happening?
# How good the prediction is? Can we rely on this analytics?

train.info()
train.isnull().sum()
test.isnull().sum()
train = train.dropna()
test = test.dropna()
train.head()


# Identify Categorical vs Numerical features first in training data set

train_cat = train.select_dtypes(exclude=['number', 'bool_', 'float_'])
train_cat.head()

train_num = train.select_dtypes(exclude=['bool_', 'object_'])
train_num.head()

train.hist()

train_cat.columns

sns.countplot(x='gender', data=train_cat)
sns.countplot(x='Partner', data=train_cat)
sns.countplot(x='Dependents', data=train_cat)
sns.countplot(x='PhoneService', data=train_cat)
sns.countplot(x='MultipleLines', data=train_cat)
sns.countplot(x='InternetService', data=train_cat)
sns.countplot(x='OnlineSecurity', data=train_cat)
sns.countplot(x='OnlineBackup', data=train_cat)
sns.countplot(x='DeviceProtection', data=train_cat)
sns.countplot(x='TechSupport', data=train_cat)
sns.countplot(x='StreamingTV', data=train_cat)
sns.countplot(x='StreamingMovies', data=train_cat)
sns.countplot(x='Contract', data=train_cat)
sns.countplot(x='PaperlessBilling', data=train_cat)
sns.countplot(x='PaymentMethod', data=train_cat)
sns.countplot(x='Churn', data=train_cat)


# Creating a filter where churn = Yes

is_churn = train['Churn'] == 'Yes'
y_true = train.Churn[is_churn]

print('We are interested in the yes class, but it has less data. Yes = ' + str((y_true.shape[0]/train.shape[0])*100))

# Convert categorical data into numerical data

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


train['gender'] = label_encoder.fit_transform(train['gender'])
train['Partner'] = label_encoder.fit_transform(train['Partner'])
train['Dependents'] = label_encoder.fit_transform(train['Dependents'])
train['PhoneService'] = label_encoder.fit_transform(train['PhoneService'])
train['MultipleLines'] = label_encoder.fit_transform(train['MultipleLines'])
train['InternetService'] = label_encoder.fit_transform(train['InternetService'])
train['OnlineSecurity'] = label_encoder.fit_transform(train['OnlineSecurity'])
train['OnlineBackup'] = label_encoder.fit_transform(train['OnlineBackup'])
train['DeviceProtection'] = label_encoder.fit_transform(train['DeviceProtection'])
train['TechSupport'] = label_encoder.fit_transform(train['TechSupport'])
train['StreamingTV'] = label_encoder.fit_transform(train['StreamingTV'])
train['StreamingMovies'] = label_encoder.fit_transform(train['StreamingMovies'])
train['Contract'] = label_encoder.fit_transform(train['Contract'])
train['PaperlessBilling'] = label_encoder.fit_transform(train['PaperlessBilling'])
train['PaymentMethod'] = label_encoder.fit_transform(train['PaymentMethod'])

print(train.dtypes)

# set X and y

y = train['Churn']
y.size
y = y.replace({'No': 0, 'Yes': 1})
y

X = train.drop(['customerID','Churn'],axis=1)
X.isnull().sum()


# Standardizing the data

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)


# Setting up Stratified K fold to handle skewed data

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfold = StratifiedKFold(n_splits = 20, random_state=51)


# Implement Gradient Boost Classifier

from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(random_state=51)

for train_index, test_index in skfold.split(X, y):
    clone_clf = clone(gb_clf)
    X_train_fold = X[train_index]
    y_train_fold = y.iloc[train_index]
    X_test_fold = X[test_index]
    y_test_fold = y.iloc[test_index]
    clone_clf.fit(X_train_fold,y_train_fold)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print('Result for Gradient Boost Classifier: ', n_correct/len(y_pred))
    

# Implement Support Vector Machines (SVM)

from sklearn.svm import SVC
svc_clf = SVC(random_state = 51)

for train_index, test_index in skfold.split(X, y):
    clone_clf = clone(svc_clf)
    X_train_fold = X[train_index]
    y_train_fold = y.iloc[train_index]
    X_test_fold = X[test_index]
    y_test_fold = y.iloc[test_index]
    clone_clf.fit(X_train_fold,y_train_fold)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print('Result for SVM: ', n_correct/len(y_pred))


# Implement Random Forest Classifier
    
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state = 51)

for train_index, test_index in skfold.split(X, y):
    clone_clf = clone(rf_clf)
    X_train_fold = X[train_index]
    y_train_fold = y.iloc[train_index]
    X_test_fold = X[test_index]
    y_test_fold = y.iloc[test_index]
    clone_clf.fit(X_train_fold,y_train_fold)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print('Result for Random Forest: ', n_correct/len(y_pred))


# Implement K nearest neighbor Classifier
    
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()

for train_index, test_index in skfold.split(X, y):
    clone_clf = clone(knn_clf)
    X_train_fold = X[train_index]
    y_train_fold = y.iloc[train_index]
    X_test_fold = X[test_index]
    y_test_fold = y.iloc[test_index]
    clone_clf.fit(X_train_fold,y_train_fold)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print('Result for K nearest neighbour: ', n_correct/len(y_pred))
    

# Implement Logistic Regression Classifier
    
from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression(random_state = 51)

for train_index, test_index in skfold.split(X, y):
    clone_clf = clone(log_clf)
    X_train_fold = X[train_index]
    y_train_fold = y.iloc[train_index]
    X_test_fold = X[test_index]
    y_test_fold = y.iloc[test_index]
    clone_clf.fit(X_train_fold,y_train_fold)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print('Result for Random Forest: ', n_correct/len(y_pred))


# Precision and recall

from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(gb_clf, X, y, cv = 10)

from sklearn.metrics import confusion_matrix    
confusion_matrix(y,y_pred)

# PRECISION = TP/(TP+FP)
# RECALL = TP/(TP+FN)

from sklearn.metrics import precision_score, recall_score
print('The precision is :', precision_score(y,y_pred))
print('The recall is :', recall_score(y, y_pred))

# Changing threshold

y_score = cross_val_predict(gb_clf, X, y, cv=10, method='decision_function')

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y, y_score)

# Plot Precision-Recall curve
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0,1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# Precision against recall
plt.plot(precisions, recalls)
plt.xlabel('Recall')
plt.ylabel('Precision')

y_new_scores = (y_score > -0.3)
print('New precision is :', precision_score(y, y_new_scores))
print('New recall is :', recall_score(y,y_new_scores))


# ROC (receiver operating characteristic)

from sklearn.metrics import roc_curve
fpr,tpr,thresholds = roc_curve(y,y_score)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth = 2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()