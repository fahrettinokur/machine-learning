# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2

resim=cv2.imread("indir.jpg",0)
cv2.imshow("resimler resmi",resim)
cv2.waitKey(0)
cv2.destroyAllWindows()

Rz=450
Vz=0
V=np.array([1,2,3,4,5,6,7,8,9,10])

for V1 in V:
    if V1>=5:
        Rz=80
        
    Is=(V1-Vz)/(100+Rz)
    plt.scatter(V1,Is)
plt.xlabel("Kaynak voltajı")
plt.ylabel("Is akımı")
plt.title("Is akımının bulunması")