from clustering.kmedoids import kmedoids
from clustering.MAE_RMSE import MAE_RMSE, createDictTestMovies
from dataset import DataSet, load_data
from preprocess import mean, svd_uu_100k
import numpy as np
import json
from bso import BSO

def bso_eval(meds, dist_mat, usage_mat, test_dict):
    m = np.array(np.where(meds == 1))[0]
    #print(meds)
    kmed = kmedoids(dist_mat, m, data_type='distance_matrix', ccore=False)
    kmed.process()
    metrics = MAE_RMSE(usage_mat, dist_mat, kmed.get_clusters(), test_dict)
    return metrics

'''
data = DataSet(
    '/home/imad/Desktop/PFE/db/ml-100k/ua.base',
    '/home/imad/Desktop/PFE/db/ml-100k/u.user',
    '/home/imad/Desktop/PFE/db/ml-100k/u.item'
)

results = {}

print("#FC user user | mean user | k-medoids")
results.update({'fc_user_user_ua': []})
usage_matrix = mean(data.get_usage_matrix())
distance_matrix = load_data('./distance_matrices/fc_uu_ua_dist_mat.csv')
test_dict = createDictTestMovies("/home/imad/Desktop/PFE/db/ml-100k/ua.test")
'''
'''
for med_count in range(10, 10, 5):
    initial_medoids = np.random.choice(
        [i for i in range(len(usage_matrix))],
        med_count
    )
    #meds = [int(x) for x in initial_medoids]
    #bso = BSO(med_count, lambda x: bso_eval(x, distance_matrix, usage_matrix, test_dict))
    kmed = kmedoids(distance_matrix,
        initial_medoids,
        data_type='distance_matrix',
        ccore=False
    )
    kmed.process()
    mae, rmse, prec, rec = MAE_RMSE(usage_matrix, distance_matrix, kmed.get_clusters(), '/home/imad/Desktop/PFE/db/ml-100k/ua.test')
    #print(meds)
    print(mae, rmse)
    results['fc_user_user_ua'].append({'meds_count': med_count, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})

print("#FC user user | svd | k-medoids")
results.update({'fc_user_user_svd': []})
test_dict = createDictTestMovies("/home/imad/Desktop/PFE/db/ml-100k/ua.test")
usage_matrix = svd_uu_100k(data.get_usage_matrix())
distance_matrix = load_data('./distance_matrices/fc_uu_svd_dist_mat.csv')


for med_count in range(10, 700, 5):
    initial_medoids = np.random.choice(
        [i for i in range(len(usage_matrix))],
        med_count
    )
    #meds = [int(x) for x in initial_medoids]
    kmed = kmedoids(distance_matrix,
        initial_medoids,
        data_type='distance_matrix',
        ccore=False
    )
    kmed.process()
    mae, rmse, prec, rec = MAE_RMSE(usage_matrix, distance_matrix, kmed.get_clusters(), '/home/imad/Desktop/PFE/db/ml-100k/ua.test')
    #print(meds)
    print(mae, rmse)
    results['fc_user_user_svd'].append({'meds_count': med_count, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})

json.dump(results, open('./results/fc_kmedoids.json', 'w'))

for med_count in range(10, 100, 5):
    bso = BSO(len(usage_matrix), med_count, lambda x: bso_eval(x, distance_matrix, usage_matrix, test_dict))
    (mae, rmse, prec, rec), s = bso.run(int(med_count/5))
    print(mae, rmse, prec, rec)
    results['fc_user_user_svd'].append({'meds_count': med_count, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})
    json.dump(results, open('./results/fc_kmedoids_bso.json', 'w'))
'''