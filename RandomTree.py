# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


data=pd.read_csv("positions.csv")

Level=data.iloc[:,1].values.reshape(-1,1)
Salary=data.iloc[:,2].values

regression=RandomForestRegressor(n_estimators=10,random_state=0)#böyle yaparsan sürekli aynı değer
#döndürür random_state değiştirirsen döndürecek değer değişir ama gene sabit kalır
regression.fit(Level,Salary)
print(regression.predict(np.array([8.3]).reshape(-1,1)))

x=np.arange(min(Level),max(Level),0.001).reshape(-1,1)
plt.scatter(Level,Salary,color="black")
plt.plot(x,regression.predict(x),color="red")
plt.show()