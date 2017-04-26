import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.info()

print(train.describe())

# show the overall survival rate (38.38), as the standard when choosing the fts
print('Overall Survival Rate:',train['Survived'].mean())

# get_dummies function
def dummies(col,train,test):
    train_dum = pd.get_dummies(train[col])
    test_dum = pd.get_dummies(test[col])
    train = pd.concat([train, train_dum], axis=1)
    test = pd.concat([test,test_dum],axis=1)
    train.drop(col,axis=1,inplace=True)
    test.drop(col,axis=1,inplace=True)
    return train, test

# get rid of the useless cols
dropping = ['PassengerId', 'Name', 'Ticket']
train.drop(dropping,axis=1, inplace=True)
test.drop(dropping,axis=1, inplace=True)

#pclass
# ensure no na contained
print(train.Pclass.value_counts(dropna=False))
sns.factorplot('Pclass', 'Survived',data=train, order=[1,2,3])
# according to the graph, we found there are huge differences between
# each pclass group. keep the ft
train, test = dummies('Pclass', train, test)

# sex
print(train.Sex.value_counts(dropna=False))
sns.factorplot('Sex','Survived', data=train)
# female survival rate is way better than the male
train, test = dummies('Sex', train, test)
# cos the male survival rate is so low, delete the male col
train.drop('male',axis=1,inplace=True)
test.drop('male',axis=1,inplace=True)