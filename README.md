<br>Makine Öğrenme Aşamaları 
<br>1-Veri setini yükle
<br>2-Kayıp değerleri doldurmak
<br>3-Veriyi "öğrenme" ve "test" olmak üzere 2'ye böl
<br>4-Uygun modeli belirle
<br>5-Seçilen modele göre öğrenmeye başla
<br>6-Tahmin yap
<br>7-Doğruluğunu kontrol et

<br>Veri Önişleme Nedir?
<br>Veri setini, model belirlemek için hazır hale getirmektir.
<br>1-Veri setindeki eksik değerleri tamamlamak
<br>2-Kategorileri numaralandırmak
<br>3-Özellikleri ölçeklendirme
<br>4-Veri setini öğrenme ve test şeklinde ikiye bölme
<br>5-Veriyi görselleştirme

<br><br>
# ai_sklearn_load_iris

# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
<br>from sklearn.datasets import load_iris 
<br>from sklearn.model_selection import train_test_split 
<br>iris_dataset = load_iris() 
<br>x_ogren,x_test,y_ogren,y_test = train_test_split(iris_dataset['data'],iris_dataset['target']) 
 
<br>#print(iris_dataset); 
<br>#print(x_ogren.shape); 
<br>#print(x_test.shape); 
<br>#print(y_ogren.shape); 
<br>#print(y_test.shape); 

<br>#uygun model seçilir
<br>from sklearn.neighbors import KNeighborsClassifier
<br>knn = KNeighborsClassifier(n_neighbors=1);

<br>#öğrenme
<br>knn.fit(x_ogren,y_ogren);
<br>print(knn);

<br>#tahmin
<br>x_yeni = [[3.5,2.1,3.4,1.2]];
<br>tahmin= knn.predict(x_yeni);
<br>print(tahmin);

<br>#dogruluk
<br>dogruluk = knn.predict(x_test)
<br>print(dogruluk);

<br>import numpy as nm
<br>print(nm.mean(dogruluk==y_test)*100);
