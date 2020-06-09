import csv
import json
import sys
from dataset import DataSet, load_data, dump_array
import pickle
import numpy as np

if __name__ == "__main__":
    '''
    res = json.load(open('./results/hyb_sem_based_fc_2_knn.json'))
    for key in res:
        res_csv = csv.writer(open(f'./results/csv/{key}_2_knn.csv', 'w'), delimiter=',')
        res_csv.writerow(['k', 'mae', 'rmse', 'prec', 'recall'])
        for test in res[key]:
            l = list(test.values())
            res_csv.writerow(l)
    
    '''
    ratings = load_data('../../db/epinions/epinions_new.data', sep="\t").astype('int')
    print(ratings)
    
    
