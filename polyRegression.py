# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


data =pd.read_csv("positions.csv")
print(data.columns)

Level=data.iloc[:,1].values.reshape(-1,1)
Salary=data.iloc[:,2].values.reshape(-1,1)

regression=LinearRegression()
regression.fit(Level,Salary)

tahmin=regression.predict(np.array([8.3]).reshape(-1,1))


polyregression = PolynomialFeatures(degree = 4) #2 de yapılır ama düzğün çıkmıyor
Levelpoly=polyregression.fit_transform(Level)
regression2=LinearRegression()
tahmin2=regression2.fit(Levelpoly,Salary)

tahmin2=regression2.predict(polyregression.fit_transform(np.array([8.3]).reshape(-1,1)))

plt.scatter(Level,Salary,color="red")
plt.plot(Level,regression.predict(Level),color="blue")
plt.plot(Level,regression2.predict(Levelpoly))
plt.show()
