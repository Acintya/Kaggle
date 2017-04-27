# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 23:14:22 2017

@author: VWKLZUL
"""

from __future__ import print_function

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
import numpy as np

print(__doc__)

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
labels = train_data['Survived']
train_data.drop(['Survived'], axis = 1, inplace = True)

dropping = ['PassengerId', 'Name', 'Ticket']
train_data.drop(dropping, axis = 1, inplace = True)
test_data.drop(dropping, axis = 1, inplace = True)


# get_dummies function
def dummies(col,train,test):
    train_dum = pd.get_dummies(train[col])
    test_dum = pd.get_dummies(test[col])
    train = pd.concat([train, train_dum], axis=1)
    test = pd.concat([test,test_dum],axis=1)
    train.drop(col,axis=1,inplace=True)
    test.drop(col,axis=1,inplace=True)
    return train, test

train_data, test_data = dummies('Pclass', train_data, test_data)
train_data, test_data = dummies('Sex', train_data, test_data)

#age 
#dealing the missing data
nan_num = train_data['Age'].isnull().sum()
# there are 177 missing value, fill with random int
age_mean = train_data['Age'].mean()
age_std = train_data['Age'].std()
filling = np.random.randint(age_mean-age_std, age_mean+age_std, size=nan_num)
train_data['Age'][train_data['Age'].isnull()==True] = filling
nan_num = train_data['Age'].isnull().sum()

# dealing the missing val in test
nan_num = test_data['Age'].isnull().sum()
# 86 null
age_mean = test_data['Age'].mean()
age_std = test_data['Age'].std()
filling = np.random.randint(age_mean-age_std,age_mean+age_std,size=nan_num)
test_data['Age'][test_data['Age'].isnull()==True]=filling

train_data['family'] = train_data['SibSp'] + train_data['Parch']
test_data['family'] = test_data['SibSp'] + test_data['Parch']

train_data.drop(['SibSp','Parch'],axis=1,inplace=True)
test_data.drop(['SibSp','Parch'],axis=1,inplace=True)

#according to the plot, smaller fare has higher survival rate, keep it
#dealing the null val in test
#test_data['Fare'].fillna(test_data['Fare'].median(),inplace=True)

train_data.drop('Cabin',axis=1,inplace=True)
test_data.drop('Cabin',axis=1,inplace=True)

train_data['Embarked'].fillna('S',inplace=True)
# c has higher survival rate, drop the other two
train_data,test_data = dummies('Embarked',train_data,test_data)


#------------------------Training------------------
# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(train_data)
X = train_data
y = labels

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()