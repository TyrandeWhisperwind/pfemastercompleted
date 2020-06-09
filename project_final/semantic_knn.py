from clustering.knn import KNN, createDictTestMovies
from dataset import DataSet, load_data
from preprocess import mean, svd_uu_100k
import json


def main():
    
    data = DataSet(
        '/home/imad/Desktop/PFE/db/ml-100k/ua.base',
        '/home/imad/Desktop/PFE/db/ml-100k/u.user',
        '/home/imad/Desktop/PFE/db/ml-100k/u.item'
    )

    movie_dict = createDictTestMovies('/home/imad/Desktop/PFE/db/ml-100k/ua.test')
    results = {}
    '''
    print("#Semantic user user | mean user | k-NN")
    results.update({'sem_user_user_ua': []})
    usage_matrix = mean(data.get_usage_matrix())
    distance_matrix = load_data('./distance_matrices/sem_uu_ua_dist_mat.csv')

    for k in range(10, 450, 5):
        knn = KNN(k, distance_matrix, usage_matrix)
        knn.process(movie_dict)
        mae, rmse, pres, rec = knn.mae_rmse()
        print(k)
        results['sem_user_user_ua'].append({'k': k, 'mae': mae, 'rmse': rmse, 'pres': pres, 'rec':rec})
    
    print("#Semantic user user | svd | k-NN")
    results.update({'sem_user_user_svd': []})
    usage_matrix = svd_uu_100k(data.get_usage_matrix())
    distance_matrix = load_data('./distance_matrices/sem_uu_svd_dist_mat.csv')

    for k in range(10, 450, 5):
        knn = KNN(k, distance_matrix, usage_matrix)
        knn.process(movie_dict)
        mae, rmse, pres, rec = knn.mae_rmse()
        print(k, mae, rmse, pres, rec)
        results['sem_user_user_svd'].append({'k': k, 'mae': mae, 'rmse': rmse, 'pres': pres, 'rec': rec})

    json.dump(results, open('./results/semantic_knn.json', 'w'))
'''

if __name__ == "__main__":
    main()