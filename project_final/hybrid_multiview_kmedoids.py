from preprocess import mean, svd_uu_100k
from dataset import DataSet, load_data
from hybridationMultiview import multiView, CalculePredection, createDictTestMovies
import numpy as np
import json

data = DataSet(
    '/home/imad/Desktop/PFE/db/ml-100k/ua.base',
    '/home/imad/Desktop/PFE/db/ml-100k/u.user',
    '/home/imad/Desktop/PFE/db/ml-100k/u.item'
)

results = {}

movie_dict = createDictTestMovies('/home/imad/Desktop/PFE/db/ml-100k/ua.test')

print("#Multiview user user | mean user | k-medoids")
results.update({'multiview_user_user_ua': []})
usage_matrix = mean(data.get_usage_matrix())
fc_distance_matrix = load_data('./distance_matrices/fc_uu_ua_dist_mat.csv')
sem_distance_matrix = load_data('./distance_matrices/sem_uu_ua_dist_mat.csv')


for med_count in range(10, 700, 5):
    initial_medoids = np.random.choice(
        [i for i in range(len(usage_matrix))],
        med_count
    )
    #meds = [int(x) for x in initial_medoids]
    #print(initial_medoids)
    clusters = multiView(initial_medoids, fc_distance_matrix, sem_distance_matrix, 100, 5)
    mae, rmse, prec, rec = CalculePredection(usage_matrix, sem_distance_matrix, fc_distance_matrix, movie_dict, clusters)
    #print(meds)
    print(mae, rmse, prec, rec)
    results['multiview_user_user_ua'].append({'meds_count': med_count, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})

print("#Multiview user user | svd | k-medoids")
results.update({'multiview_user_user_svd': []})
usage_matrix = svd_uu_100k(data.get_usage_matrix())
fc_distance_matrix = load_data('./distance_matrices/fc_uu_svd_dist_mat.csv')
sem_distance_matrix = load_data('./distance_matrices/sem_uu_svd_dist_mat.csv')

for med_count in range(10, 700,5):
    initial_medoids = np.random.choice(
        [i for i in range(len(usage_matrix))],
        med_count
    )
    #meds = [int(x) for x in initial_medoids]
    clusters = multiView(initial_medoids, fc_distance_matrix, sem_distance_matrix, 100, 5)
    mae, rmse, prec, rec = CalculePredection(usage_matrix, sem_distance_matrix, fc_distance_matrix, movie_dict, clusters)
    #print(meds)
    print(mae, rmse, prec, rec)
    results['multiview_user_user_svd'].append({'meds_count': med_count, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})

json.dump(results, open('./results/multiview_kmedoids.json', 'w'))
