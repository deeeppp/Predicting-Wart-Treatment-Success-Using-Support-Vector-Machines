#!/usr/bin/env python
# coding: utf-8

# In[40]:


# SVM 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.decomposition import PCA 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import plotly.express as pl
data=pd.read_csv(r"E:\machine learning\svmDataset.csv")
df=pd.DataFrame(data)
data=df.iloc[:,:-2]
print("short summary of the dataset \n \n")
print(df.head())

lab=LabelEncoder()
df['methods']=lab.fit_transform(df['methods'])


x=df.iloc[:,:-2]
y=df.iloc[:,-1]
print("\n \n preview of independent variable data: \n ")
print(x.head())
print("\n \n preview of dependent variable data: \n")
print(y.head())

mm=MinMaxScaler()
x=mm.fit_transform(x)
# print(x)

x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2)
sv=svm.SVC(kernel='rbf',C=50,gamma=100).fit(x_tr,y_tr)

pr=sv.predict(x_te)
print("\n \n predected value is: \n")
pr


# In[45]:


from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_te,pr))
acc = accuracy_score(y_te,pr)
print("The accuracy is {}".format(acc))
sum=0
le=pr.shape

for i in range(le[0]):
    sum=sum+abs((y_te.values[i]-pr[i]))
#     print(sum)
er=sum/le[0]
print(er)

