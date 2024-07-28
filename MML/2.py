import numpy as np
import pandas as pd

data = pd.read_csv('enjoysport.csv')
concepts = np.array(data.iloc[:, :-1])
print(concepts)
target = np.array(data.iloc[:, -1])
print(target)

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            specific_h = ['?' if specific_h[x] != h[x] else specific_h[x] for x in range(len(specific_h))]
        elif target[i] == "no":
            general_h = [['?' if specific_h[x] == h[x] else specific_h[x] for x in range(len(specific_h))] if specific_h[x] != h[x] else general_h[x] for x in range(len(specific_h))]

    general_h = [h for h in general_h if h != ['?' for _ in range(len(specific_h))]]
    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print(s_final)
print(f'Genaral hypothesus :{g_final}')
