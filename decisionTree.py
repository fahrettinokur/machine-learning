# -*- coding: utf-8 -*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data=pd.read_csv("positions.csv")

Level=data.iloc[:,1].values.reshape(-1,1)
Salary=data.iloc[:,2].values.reshape(-1,1)

regression=DecisionTreeRegressor()
regression.fit(Level,Salary)
print(regression.predict(np.array([5.6]).reshape(-1,1)))


plt.scatter(Level,Salary,color="black")
x=np.arange(min(Level),max(Level),0.000001).reshape(-1,1)#buradaki amaç az verimiz vardi biz veriyi artırdık ve 
#sistemin nasıl çalıştığını gördü
#eger böyle yaparsan hiçbir halt göremesin
#plt.plot(Level,regression.predict(Level))  sadece düz bir çizgi
plt.plot(x,regression.predict(x),color="red")
plt.show()







