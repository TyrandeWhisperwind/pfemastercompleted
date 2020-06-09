from clustering.knn import KNN, createDictTestMovies
from dataset import DataSet, load_data
from preprocess import mean, svd_uu_100k
import json

data = DataSet(
    '/home/imad/Desktop/PFE/db/ml-100k/ua.base',
    '/home/imad/Desktop/PFE/db/ml-100k/u.user',
    '/home/imad/Desktop/PFE/db/ml-100k/u.item'
)

print("#FC user user | user average | k-NN")
usage_matrix = mean(data.get_usage_matrix())
#distance_matrix = joblib.load('./distance_matrices/fc_uu_ua_dist_mat.sav')
#distance_matrix = pearson_distance(usage_matrix)
distance_matrix = load_data('./distance_matrices/fc_uu_ua_dist_mat.csv')

movie_dict = createDictTestMovies('/home/imad/Desktop/PFE/db/ml-100k/ua.test')
results = {"fc_user_user_ua": [], "fc_user_user_svd": []}
for k in range(10, 700, 5):
    knn = KNN(k, distance_matrix, usage_matrix)
    knn.process(movie_dict)
    mae, rmse, prec, rec = knn.mae_rmse()
    results["fc_user_user_ua"].append({'k': k, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})
    print(f"k = {k}", knn.mae_rmse())

usage_matrix = svd_uu_100k(data.get_usage_matrix())
distance_matrix = load_data('./distance_matrices/fc_uu_svd_dist_mat.csv')

for k in range(10, 700, 5):
    knn = KNN(k, distance_matrix, usage_matrix)
    knn.process(movie_dict)
    mae, rmse, prec, rec = knn.mae_rmse()
    results["fc_user_user_svd"].append({'k': k, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})
    print(f"k = {k}", knn.mae_rmse())

json.dump(results, open('./results/fc_knn.json', 'w'))