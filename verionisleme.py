# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

dataset = pd.read_csv("Data.csv")
print(dataset)
print(type(dataset))

print("")

print(dataset.values)
print(type(dataset.values))

print(dataset.values[0])
print(dataset.ndim)
print(dataset.size)
print(dataset.shape)
print(dataset.values.shape)
print(dataset.dtypes)

values = dataset.values
print(values[0,0])
print(values[0:2])
#print(values([:2])
print(values[:,1])
print(values[1::2])

# Veriyi X ve Y olarak ikiye bölme
X = dataset.iloc[:,:3].values
y = dataset.iloc[:,-1].values 
print(X)
print(y)

# Kayıp Değerleri Doldurma
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:]) 
X[:,1:] = imputer.transform(X[:,1:])
print(X[:,1:])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
y= le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_ogren,x_test,y_ogren,y_test = train_test_split(X,y,test_size=0.2)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_ogren[:,3:] = scaler.fit_transform(x_ogren[:,3:])
x_test[:,3:] = scaler.transform(x_test[:,3:])
