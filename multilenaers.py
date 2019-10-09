# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


data=pd.read_csv("insurance.csv")

## y ekseni
expenses=data.expenses.values.reshape(-1,1)

##x ekseni
agebmi=data.iloc[:,[0,2]].values

regression =LinearRegression()
regression.fit(agebmi,expenses)


print(regression.predict(np.array([[20,20],[20,21],[20,22],[20,23],[20,24]]).reshape(-1,2)))
print(type(agebmi[0]))
print(r2_score(expenses,regression.predict(agebmi)))


