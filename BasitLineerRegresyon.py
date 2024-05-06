# BASİT LİNEER REGRESYON 
# instagram: @bilgisayar.bilimi
# Youtube: https://www.youtube.com/channel/UCpO_zXh4c9n43-4cRnvJptg

## Gerekli kütüphanelerin eklenmesi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Veri setinin eklenmesi
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # özellikler / bağımsız değişken
y = dataset.iloc[:, -1].values # bağımlı değişken

## Veri setinin Öğrenme ve Test verisi olmak üzere ikiye bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state=0)

## Basit Lineer Regresyon modelinin Öğrenme verisinde öğrenmesini gerçekleştirmesi
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # Öğrenme aşaması

## Test verisi üzerinde tahminlerin yapılması
y_pred = regressor.predict(X_test)
#y_pred : tahmin edilen maaşlar
#y_test : gerçek maaşlar

## Öğrenme sonuçlarının görselleştirilmesi
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Maaş ve Deneyim (Öğrenme verisi)')
plt.xlabel('Deneyim')
plt.ylabel('Maaş')
plt.show()

## Test sonuçlarının görselleştirilmesi
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue') #  burada regresyon çizgisini değiştirmememiz gerekiyor yani plt.plot(X_test,y_test ..) yerine parametre olarak X_train de kalmalıdır çünkü regresyon çizgisi ünik olup öğrenme verisi aracılıyla çizilir.
plt.title('Maaş ve Deneyim(Test verisi)')
plt.xlabel('Deneyim')
plt.ylabel('Maaş')
plt.show()

## Tahmin yapmak
# Örneğin 12 yıllık tecrübesi olan birinin alacağı maaşı tahmin etmek
print(regressor.predict([[12]]))
# Tahmin fonksiyonu her zaman iki boyutlu (2D) dizi kabul eder
# 12 -> skaler, [12] -> 1D dizi , [[12]] -> 2D dizi

## Basit lineer regresyon formülünün bulunması
print(regressor.coef_) # b1
print(regressor.intercept_) # b0
# Maaş = 9345.94 x Deneyim + 26816.19