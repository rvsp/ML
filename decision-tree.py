#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 18:41:39 2019

@author: venkat
"""
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

data = pd.read_csv("https://raw.githubusercontent.com/trainervenkat/MDUSIT/master/datasets/iris_data.csv")
#data = datasets.load_iris()
print(data.head())
features = data[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
targets = data.Class 

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=2)

'''Gini index approach is faster since we don't have to compute 
log values whcih are expensive'''
#model = DecisionTreeClassifier(criterion='gini')

model = DecisionTreeClassifier(criterion='entropy')

model.fitted = model.fit(feature_train, target_train)
model.predictions = model.fitted.predict(feature_test)

print('confusion matrix: ',confusion_matrix(target_test, model.predictions))
print('accuracy score: ',accuracy_score(target_test, model.predictions))


#validate preformance with cross validation
scores=cross_val_score(model,features,targets,cv=5)
print("cross-validation: %3f"%(scores.mean()))