from clustering.knn import KNNMultiview, createDictTestMovies
from dataset import DataSet, load_data
from preprocess import mean, svd_uu_100k
import json

data = DataSet(
    '/home/imad/Desktop/PFE/db/ml-100k/ua.base',
    '/home/imad/Desktop/PFE/db/ml-100k/u.user',
    '/home/imad/Desktop/PFE/db/ml-100k/u.item'
)

results = {}
movie_dict = createDictTestMovies('/home/imad/Desktop/PFE/db/ml-100k/ua.test')
'''
print("#Multiview user user | user average | k-NN")
results.update({'multiview_user_user_ua': []})
usage_matrix = mean(data.get_usage_matrix())
#distance_matrix = joblib.load('./distance_matrices/fc_uu_ua_dist_mat.sav')
#distance_matrix = pearson_distance(usage_matrix)
distance_matrix1 = load_data('./distance_matrices/fc_uu_ua_dist_mat.csv')
distance_matrix2 = load_data('./distance_matrices/sem_uu_ua_dist_mat.csv')


for k in range(10, 700, 5):
    knn_mul = KNNMultiview(k, distance_matrix1, distance_matrix2, usage_matrix)
    knn_mul.process(movie_dict)
    print(knn_mul.mae_rmse())
    mae, rmse, prec, rec = knn_mul.mae_rmse()
    results['multiview_user_user_ua'].append({'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})

print("#Multiview user user | svd | k-NN")
results.update({'multiview_user_user_svd': []})
usage_matrix = svd_uu_100k(data.get_usage_matrix())

distance_matrix1 = load_data('./distance_matrices/fc_uu_svd_dist_mat.csv')
distance_matrix2 = load_data('./distance_matrices/sem_uu_svd_dist_mat.csv')

#movie_dict = createDictTestMovies('/home/imad/Desktop/PFE/db/ml-100k/ua.test')
for k in range(10, 700, 5):
    knn_mul = KNNMultiview(k, distance_matrix1, distance_matrix2, usage_matrix)
    knn_mul.process(movie_dict)
    print(knn_mul.mae_rmse())
    mae, rmse, prec, rec = knn_mul.mae_rmse()
    results['multiview_user_user_svd'].append({'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})

json.dump(results, open('./results/multiview_knn.json', 'w'))

'''