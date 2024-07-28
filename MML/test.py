import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


data=pd.read_csv('enjoysport.csv')
counter=np.array(data.iloc[:,:-1])
target=np.array(data.iloc[:,-1])

def learn(counter,target):
    shrink=counter[0].copy()
    general=[['?' for _ in range(len(shrink))]for _ in range(len(shrink))]
    for i,h in enumerate(counter):
        if target[i]=='yes':
            shrink=["?" if shrink[x]!=h[x] else shrink[x] for x in range(len(shrink)) ] 
        if target[i]=='no':
            general=[["?" if shrink[x]==h[x] else shrink[x] for x in range(len(shrink)) ]if shrink[i]!=h[x] else shrink[x] for x in range(len(shrink))]
            
    general=[h for h in general if h!=['?' for _ in range(len(shrink))]]
    return shrink,general

shrink,general=learn(counter,target)
print(shrink)
print(general)
