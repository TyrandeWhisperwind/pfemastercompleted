from surprise import SVD, Dataset, Reader, accuracy
import pandas as pd
import numpy as np
import pickle

def array_average(arr):
    s = np.sum(arr)
    if s == 0: return 0
    return s/np.count_nonzero(arr)

def mean(arr: np.array, axis=1):
    return pd.DataFrame(arr).apply(
        lambda row: row.replace(0, array_average(row)),
        axis=axis
    ).values

def svd_uu_100k(arr: np.array, model_path):
    model = pickle.load(open(model_path, 'rb'))
    
    result = np.array(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] == 0: result[i, j] = model.predict(i+1, j+1).est
            else: result[i, j] = arr[i, j]
            
    return result

def svd_ii_100k(arr: np.array):
    model = pickle.load(open('./svd_ml_100k.sav', 'rb'))
    
    result = np.array(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] == 0: result[i, j] = model.predict(j+1, i+1).est
            else: result[i, j] = arr[i, j]
            
    return result

def gnb_uu_100k(arr: np.array):
    model = pickle.load(open('./gnb_ml_100k.sav', 'rb'))
    
    result = np.array(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] == 0: result[i, j] = model.predict([[i+1, j+1]])[0]
            else: result[i, j] = arr[i, j]
            
    return result

def gnb_ii_100k(arr: np.array):
    model = pickle.load(open('./gnb_ml_100k.sav', 'rb'))
    
    result = np.array(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] == 0: result[i, j] = model.predict([[j+1, i+1]])[0]
            else: result[i, j] = arr[i, j]
            
    return result

if __name__ == "__main__":
    a = np.array(
        [
            [2,4,0,2],
            [8,4,0,0]
        ]
    )

    print(mean(a, axis=0))