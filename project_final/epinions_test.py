from dataset import DataSet, load_data
from clustering.knn import KNN, KNNMultiview, createDictTestMovies
from clustering.MAE_RMSE import MAE_RMSE, createDictTestMovies as createDictTestMovies2
from clustering.kmedoids import kmedoids
import pickle
import json
import numpy as np

from bso import BSO
from fc_kmedoids import bso_eval

def main_knn():
    data = pickle.load(open('../../db/epinions/epinions-svd-new.sav', 'rb'))
    test_dict = createDictTestMovies('../../db/epinions/ea.test')
    print(test_dict)
    results = {}

    print("Semantic User/User SVD KNN")
    results.update({'sem_uu_svd': []})
    usage_matrix = data.get_usage_matrix()
    dist_mat = load_data('distance_matrices/epinions/sem_user_user_wp_svd_2_dist_mat.csv')
    for k in range(5, len(dist_mat)+1):
        knn = KNN(k, dist_mat, usage_matrix)
        knn.process(test_dict)
        mae, rmse, prec, rec = knn.mae_rmse()
        print(k, mae, rmse, prec, rec)
        results['sem_uu_svd'].append({'k': k, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        json.dump(results, open('results/epinions/epinions_svd_knn.json', 'w'))

    print("FC User/User SVD KNN")
    results.update({'fc_uu_svd': []})
    dist_mat = load_data('distance_matrices/epinions/fc_user_user_svd_dist_mat.csv')
    for k in range(5, len(dist_mat)+1):
        knn = KNN(k, dist_mat, usage_matrix)
        knn.process(test_dict)
        mae, rmse, prec, rec = knn.mae_rmse()
        print(k, mae, rmse, prec, rec)
        results['fc_uu_svd'].append({'k': k, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        json.dump(results, open('results/epinions/epinions_svd_knn.json', 'w'))
    
    print("FC Based Sem User/User SVD KNN")
    results.update({'fc_based_sem_uu_svd': []})
    dist_mat = load_data('distance_matrices/epinions/fc_ii_svd_based_sem_dist_mat.csv')
    for k in range(5, len(dist_mat)+1):
        knn = KNN(k, dist_mat, usage_matrix)
        knn.process(test_dict)
        mae, rmse, prec, rec = knn.mae_rmse()
        print(k, mae, rmse, prec, rec)
        results['fc_based_sem_uu_svd'].append({'k': k, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        json.dump(results, open('results/epinions/epinions_svd_knn.json', 'w'))


def main_knn_multiview():
    data = pickle.load(open('../../db/epinions/epinions-svd-new.sav', 'rb'))
    test_dict = createDictTestMovies('../../db/epinions/ea.test')
    print(test_dict)
    results = {}

    print("Semantic User/User SVD KNN")
    results.update({'sem_uu_svd': []})
    usage_matrix = data.get_usage_matrix()
    dist_mat1 = load_data('distance_matrices/epinions/sem_user_user_wp_svd_dist_mat.csv')
    dist_mat2 = load_data('distance_matrices/epinions/fc_user_user_svd_dist_mat.csv')
    for k in range(5, len(dist_mat1)+1):
        knn = KNNMultiview(k, dist_mat1, dist_mat2, usage_matrix)
        knn.process(test_dict)
        mae, rmse, prec, rec = knn.mae_rmse()
        print(k, mae, rmse, prec, rec)
        results['sem_uu_svd'].append({'k': k, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        json.dump(results, open('results/epinions/epinions_svd_knn_multiview.json', 'w'))
    

def main_kmedoids():
    data = pickle.load(open('../../db/epinions/epinions-svd-new.sav', 'rb'))
    test_dict = createDictTestMovies2('../../db/epinions/ea.test')
    print(test_dict)
    results = {}

    print("Semantic User/User SVD K-medoids")
    results.update({'sem_uu_svd': []})
    usage_matrix = data.get_usage_matrix()
    dist_mat = load_data('distance_matrices/epinions/sem_user_user_wp_svd_dist_mat.csv')
    for med_count in range(5, round(len(dist_mat)/2)):
        meds = np.random.choice(
            [i for i in range(len(usage_matrix))],
            med_count
        )
        kmed = kmedoids(
            dist_mat,
            meds,
            data_type='distance_matrix',
            ccore=False
        )
        kmed.process()
        mae, rmse, prec, rec = MAE_RMSE(usage_matrix, dist_mat, kmed.get_clusters(), test_dict)
        print(med_count, mae, rmse, prec, rec)
        results['sem_uu_svd'].append({'med_count': med_count, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        json.dump(results, open('results/epinions/epinions_svd_kmedoids.json', 'w'))

    print("FC User/User SVD K-med")
    results.update({'fc_uu_svd': []})
    dist_mat = load_data('distance_matrices/epinions/fc_user_user_svd_dist_mat.csv')
    for med_count in range(5, round(len(dist_mat)/2)):
        meds = np.random.choice(
            [i for i in range(len(usage_matrix))],
            med_count
        )
        kmed = kmedoids(
            dist_mat,
            meds,
            data_type='distance_matrix',
            ccore=False
        )
        kmed.process()
        mae, rmse, prec, rec = MAE_RMSE(usage_matrix, dist_mat, kmed.get_clusters(), test_dict)
        print(med_count, mae, rmse, prec, rec)
        results['fc_uu_svd'].append({'med_count': med_count, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        json.dump(results, open('results/epinions/epinions_svd_kmedoids.json', 'w'))
    
    print("FC Based Sem User/User SVD K-med")
    results.update({'fc_based_sem_uu_svd': []})
    dist_mat = load_data('distance_matrices/epinions/fc_ii_svd_based_sem_dist_mat.csv')
    for med_count in range(5, round(len(dist_mat)/2)):
        meds = np.random.choice(
            [i for i in range(len(usage_matrix))],
            med_count
        )
        kmed = kmedoids(
            dist_mat,
            meds,
            data_type='distance_matrix',
            ccore=False
        )
        kmed.process()
        mae, rmse, prec, rec = MAE_RMSE(usage_matrix, dist_mat, kmed.get_clusters(), test_dict)
        print(med_count, mae, rmse, prec, rec)
        results['fc_based_sem_uu_svd'].append({'med_count': med_count, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        json.dump(results, open('results/epinions/epinions_svd_kmedoids.json', 'w'))

def main_kmedoids_bso():
    data = pickle.load(open('../../db/epinions/epinions-svd-new.sav', 'rb'))
    test_dict = createDictTestMovies2('../../db/epinions/ea.test')
    #print(test_dict)
    results = {}

    print("Semantic User/User SVD K-medoids")
    results.update({'sem_uu_svd': []})
    usage_matrix = data.get_usage_matrix()
    dist_mat = load_data('distance_matrices/epinions/sem_user_user_wp_svd_dist_mat.csv')
    for med_count in range(5, round(len(dist_mat)/2)):
        meds = np.random.choice(
            [i for i in range(len(usage_matrix))],
            med_count
        )
        kmed = kmedoids(
            dist_mat,
            meds,
            data_type='distance_matrix',
            ccore=False
        )
        kmed.process()
        bso = BSO(
            len(dist_mat),
            kmed.get_medoids(),
            lambda x: bso_eval(x, dist_mat, usage_matrix, test_dict),
            from_count=False
        )
        (mae, rmse, prec, rec), _ = bso.run(7)
        print(med_count, mae, rmse, prec, rec)
        results['sem_uu_svd'].append({'med_count': med_count, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        json.dump(results, open('results/epinions/epinions_svd_kmedoids_bso.json', 'w'))

    print("FC User/User SVD K-med")
    results.update({'fc_uu_svd': []})
    dist_mat = load_data('distance_matrices/epinions/fc_user_user_svd_dist_mat.csv')
    for med_count in range(5, round(len(dist_mat)/2)):
        meds = np.random.choice(
            [i for i in range(len(usage_matrix))],
            med_count
        )
        kmed = kmedoids(
            dist_mat,
            meds,
            data_type='distance_matrix',
            ccore=False
        )
        kmed.process()
        bso = BSO(
            len(dist_mat),
            kmed.get_medoids(),
            lambda x: bso_eval(x, dist_mat, usage_matrix, test_dict),
            from_count=False
        )
        (mae, rmse, prec, rec), _ = bso.run(7)
        print(med_count, mae, rmse, prec, rec)
        results['fc_uu_svd'].append({'med_count': med_count, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        json.dump(results, open('results/epinions/epinions_svd_kmedoids_bso.json', 'w'))
    
    print("FC Based Sem User/User SVD K-med")
    results.update({'fc_based_sem_uu_svd': []})
    dist_mat = load_data('distance_matrices/epinions/fc_ii_svd_based_sem_dist_mat.csv')
    for med_count in range(5, round(len(dist_mat)/2)):
        meds = np.random.choice(
            len(usage_matrix),
            med_count
        )
        kmed = kmedoids(
            dist_mat,
            meds,
            data_type='distance_matrix',
            ccore=False
        )
        kmed.process()
        bso = BSO(
            len(usage_matrix), 
            kmed.get_medoids(), 
            lambda x: bso_eval(x, dist_mat, usage_matrix, test_dict), 
            from_count=False
        )
        (mae, rmse, prec, rec), _ = bso.run(7)
        print(med_count, mae, rmse, prec, rec)
        results['fc_based_sem_uu_svd'].append({'med_count': med_count, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        json.dump(results, open('results/epinions/epinions_svd_kmedoids_bso.json', 'w'))


if __name__ == "__main__":
    main_kmedoids_bso()