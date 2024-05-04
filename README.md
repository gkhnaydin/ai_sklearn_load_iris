<br>Kaynak <b>https://www.youtube.com/watch?v=rSyDZy9lgZQ&list=PLp2P22ZlXBaa01iCUMvuXNt96G8XKNgBH&index=5&ab_channel=AlperenBayramo%C4%9Flu</b>
<br>Makine Öğrenme Aşamaları 
<br>1-Veri setini yükle
<br>2-Kayıp değerleri doldurmak
<br>3-Veriyi "öğrenme" ve "test" olmak üzere 2'ye böl
<br>4-Uygun modeli belirle
<br>5-Seçilen modele göre öğrenmeye başla
<br>6-Tahmin yap
<br>7-Doğruluğunu kontrol et

<br><b>Veri Önişleme Nedir?</b>
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

<br>
Verileri etiketlerken 2 yöntem kullanılır. LabelEncoder, OneHotEncoder
<br>LabelEncoder
<br>1-2 veya daha az sınıf sayılarında kullanılır. Örneğin veri setindeki özellik değeri "Evet"-"Hayır" ise bu şekilde kullanılabilir. Etiketler 0 ile etiket-sayisi-1 aralığında numaralandırılır. 0-Evet, 1-Hayır
<br>OneHotEncoder
<br>2den daha fazla sınıf sayılarında kullanılır.Etiketleri uniq olarak ikili sayı sisteminde numaralandırılır.Örneğin veri setindeki özellikler Almanya-Fransa-Rusya olsun:
<br>Almanya -->001
<br>Fransa  -->010
<br>Rusya   --100


<br>Python kodu
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

<br>import numpy as np
<br>import pandas as pd
<br>import matplotlib.pyplot as plt
<br>from sklearn.impute import SimpleImputer

<br>dataset = pd.read_csv("Data.csv")
<br>print(dataset)
<br>print(type(dataset))

<br>print("")

<br>print(dataset.values)
<br>print(type(dataset.values))

<br>print(dataset.values[0])
<br>print(dataset.ndim)
<br>print(dataset.size)
<br>print(dataset.shape)
<br>print(dataset.values.shape)
<br>print(dataset.dtypes)

<br>values = dataset.values
<br>print(values[0,0])
<br>print(values[0:2])
<br>#print(values([:2])
<br>print(values[:,1])
<br>print(values[1::2])

<br># Veriyi X ve Y olarak ikiye bölme
<br>X = dataset.iloc[:,:3].values
<br>y = dataset.iloc[:,-1].values 
<br>print(X)
<br>print(y)

<br># Kayıp Değerleri Doldurma
<br>imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
<br>imputer.fit(X[:,1:]) 
<br>X[:,1:] = imputer.transform(X[:,1:])
<br>print(X[:,1:])


<br>from sklearn.compose import ColumnTransformer
<br>from sklearn.preprocessing import OneHotEncoder
<br>ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
<br>X = np.array(ct.fit_transform(X))


<br>from sklearn.preprocessing import LabelEncoder
<br>le= LabelEncoder()
<br>y= le.fit_transform(y)

<br>from sklearn.model_selection import train_test_split
<br>x_ogren,x_test,y_ogren,y_test = train_test_split(X,y,test_size=0.2)

<br><b>Özellikleri Ölçekleme</b>
<br>1-Bütün özelliklerin aynı ölçekte işlem görmesini sağlar.
<br>2-Bazı özelliklerin diğer özellikleri domine etmesinden kaynaklanır.
<br>3-Bütün değerler aynı aralıkta olur.
