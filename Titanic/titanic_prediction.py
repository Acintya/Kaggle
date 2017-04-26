# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 23:27:29 2017

@author: Ling
"""

# import machine learning libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold

def modeling(clf,ft,target):
    acc = cross_val_score(clf,ft,target,cv=kf)
    acc_lst.append(acc.mean())
    return 

accuracy = []
def ml(ft,target,time):
    accuracy.append(acc_lst)

    #logisticregression
    logreg = LogisticRegression()
    modeling(logreg,ft,target)
    #RandomForest
    rf = RandomForestClassifier(n_estimators=50,min_samples_split=4,min_samples_leaf=2)
    modeling(rf,ft,target)
    #svc
    svc = SVC()
    modeling(svc,ft,target)
    #knn
    knn = KNeighborsClassifier(n_neighbors = 3)
    modeling(knn,ft,target)
    
    
    # see the coefficient
    logreg.fit(ft,target)
    feature = pd.DataFrame(ft.columns)
    feature.columns = ['Features']
    feature["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
    print(feature)
    return 