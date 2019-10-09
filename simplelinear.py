# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression #aslında bu class gibi çalışıyor
from sklearn.metrics import r2_score
data=pd.read_csv("hw_25000.csv")
#boy1=data.Height #böyle yaparsan pandasın başındaki seriler gibi kaydeder
boy=data.Height.values.reshape(-1,1)#satır ne olursa olsun ama bana sütünlü olsun 1 tane
kilo=data.Weight.values.reshape(-1,1)

regression=LinearRegression()
regression.fit(boy,kilo)
x=int(input("Boyunuzu giriniz biz size kilonuzu tahmin edelim="))
print(regression.predict(np.array([x] ).reshape(1,1) ) )




print(data.columns)

plt.scatter(data.Height,data.Weight)
t=np.arange(min(data.Height),max(data.Height)).reshape(-1,1)
plt.scatter(t,regression.predict(t),color="red")
plt.xlabel("boy")
plt.ylabel("kilo")
plt.title("Simple regression")
plt.show()


print("Bu datalara göre doğruluk yüzdesi =",r2_score(kilo,regression.predict(boy)))