# ai_sklearn_load_iris

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset = load_iris()
x_ogren,x_test,y_ogren,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'])


#print(iris_dataset);
#print(x_ogren.shape);
#print(x_test.shape);
#print(y_ogren.shape);
#print(y_test.shape);

#uygun model seçilir
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1);

#öğrenme
knn.fit(x_ogren,y_ogren);
print(knn);

#tahmin
x_yeni = [[3.5,2.1,3.4,1.2]];
tahmin= knn.predict(x_yeni);
print(tahmin);

#dogruluk
dogruluk = knn.predict(x_test)
print(dogruluk);

import numpy as nm
print(nm.mean(dogruluk==y_test)*100);
