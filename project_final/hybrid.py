from dataset import DataSet, load_data, dump_array
from clustering.knn import KNN, createDictTestMovies
from clustering.kmedoids import kmedoids
from clustering.MAE_RMSE import MAE_RMSE
from fc_user_user import fc_user_user
from preprocess import mean, svd_uu_100k
from semantic import semantic, jaccard_distance, get_sorted_item_matrix, compute_user_matrix
import numpy as np
import json

from bso import BSO

def alpha_beta_hybrid(dist_mat_1, dist_mat_2, alpha):
    return alpha*dist_mat_1 + (1 - alpha)*dist_mat_2

def semantic_based_fc_hybrid(dist_mat_sem, dist_mat_fc):
    n = len(dist_mat_fc)
    return np.array(
        [
            [
                dist_mat_fc[i, j] if dist_mat_fc[i,j] < 0.5
                else dist_mat_sem[i, j]
                for i in range(n)
            ]
            for j in range(n)
        ]
    )

def fc_based_semantic_hybrid(item_mat, usage_mat):
    sorted_item_matrix = get_sorted_item_matrix(item_mat)
    return compute_user_matrix(usage_mat, sorted_item_matrix)

def best_alpha_beta_hybrid(dist_mat_1, dist_mat_2, eval_func):
    min_mae = 1
    best_alpha = 0
    best_mat = None
    for alpha in range(1, 9):
        dist_mat = alpha_beta_hybrid(dist_mat_1, dist_mat_2, alpha/10)
        mae, rmse, prec, rec = eval_func(dist_mat)
        if mae < min_mae:
            print(mae)
            best_mat = dist_mat
            (min_mae, best_rmse, best_prec, best_rec) = (mae, rmse, prec, rec)
            best_alpha = alpha
    
    return (best_mat, min_mae, best_rmse, best_prec, best_rec, best_alpha)

def hybrid_knn_eval(k, dist_mat, usage_matrix, test_dict):
    knn = KNN(k, dist_mat, usage_matrix)
    knn.process(test_dict)

    return knn.mae_rmse()

def hybrid_kmedoids_eval(medoids, dist_mat, usage_matrix, test_file):
    kmed = kmedoids(
        dist_mat,
        medoids,
        data_type='distance_matrix',
        ccore=False
    )
    kmed.process()
    return MAE_RMSE(usage_matrix, dist_mat, kmed.get_clusters(), test_file), kmed.get_medoids()

def bso_eval(meds, dist_mat, usage_mat, test_dict):
    m = np.array(np.where(meds == 1))[0]
    #print(meds)
    kmed = kmedoids(dist_mat, m, data_type='distance_matrix', ccore=False)
    kmed.process()
    metrics = MAE_RMSE(usage_mat, dist_mat, kmed.get_clusters(), test_dict)
    return metrics


def run_test(data):
    results = {}
    results.update({'hybrid_alpha_uu_ua': []})
    usage_matrix = mean(data.get_usage_matrix())
    
    #test_dict = createDictTestMovies('/home/imad/Desktop/PFE/db/ml-100k/ua.test')

    #fc_dist_mat = load_data('./distance_matrices/fc_ii_ia_dist_mat.csv')
    #fc_ua_dist_mat = load_data('./distance_matrices/fc_ii_ua_dist_mat.csv')
    #fc_svd_dist_mat = load_data('./distance_matrices/fc_ii_svd_dist_mat.csv')
    
    
    #sem_dist_mat = load_data('./distance_matrices/sem_uu_ua_dist_mat.csv')
    fc_dist_mat = load_data('./distance_matrices/fc_uu_ua_dist_mat.csv')
    sem_dist_mat = load_data('./distance_matrices/sem_uu_ua_dist_mat.csv')

    #dist_mat, mae, rmse = best_alpha_beta_hybrid(fc_dist_mat, sem_dist_mat, lambda x: hybrid_knn_eval(152, x, usage_matrix, test_dict))
    vmin1 = np.amin(fc_dist_mat)
    vmin2 = np.amin(sem_dist_mat)
    vmin = vmin1 if vmin1 <= vmin2 else vmin2
    vmax1 = np.amax(fc_dist_mat)
    vmax2 = np.amax(sem_dist_mat)
    vmax = vmax1 if vmax1 >= vmax2 else vmax2

    step = (vmax - vmin)/10

    t = vmin
    while(t <= vmax):
        dist_mat, mae, rmse, prec, rec, alpha = best_alpha_beta_hybrid(
            fc_dist_mat,
            sem_dist_mat,
            lambda x: fc_user_user(usage_matrix, x, '/home/imad/Desktop/PFE/db/ml-100k/ua.test', threshold=t)
        )
        results['hybrid_alpha_uu_ua'].append({'threshold': t, 'alpha': alpha, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        t = t + step

    results.update({'hybrid_alpha_uu_svd': []})
    usage_matrix = svd_uu_100k(data.get_usage_matrix())

    #test_dict = createDictTestMovies('/home/imad/Desktop/PFE/db/ml-100k/ua.test')

    #fc_dist_mat = load_data('./distance_matrices/fc_ii_ia_dist_mat.csv')
    #fc_ua_dist_mat = load_data('./distance_matrices/fc_ii_ua_dist_mat.csv')
    #fc_svd_dist_mat = load_data('./distance_matrices/fc_ii_svd_dist_mat.csv')
    
    
    #sem_dist_mat = load_data('./distance_matrices/sem_uu_ua_dist_mat.csv')
    fc_dist_mat = load_data('./distance_matrices/fc_uu_svd_dist_mat.csv')
    sem_dist_mat = load_data('./distance_matrices/sem_uu_svd_dist_mat.csv')

    #dist_mat, mae, rmse = best_alpha_beta_hybrid(fc_dist_mat, sem_dist_mat, lambda x: hybrid_knn_eval(152, x, usage_matrix, test_dict))
    vmin1 = np.amin(fc_dist_mat)
    vmin2 = np.amin(sem_dist_mat)
    vmin = vmin1 if vmin1 <= vmin2 else vmin2
    vmax1 = np.amax(fc_dist_mat)
    vmax2 = np.amax(sem_dist_mat)
    vmax = vmax1 if vmax1 >= vmax2 else vmax2

    step = (vmax - vmin)/10

    t = vmin
    while(t <= vmax):
        dist_mat, mae, rmse, prec, rec, alpha = best_alpha_beta_hybrid(
            fc_dist_mat,
            sem_dist_mat,
            lambda x: fc_user_user(usage_matrix, x, '/home/imad/Desktop/PFE/db/ml-100k/ua.test', threshold=t)
        )
        results['hybrid_alpha_uu_svd'].append({'threshold': t, 'alpha': alpha, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        t = t + step

    json.dump(results, open('./results/hyb_alpha.json', 'w'))

    results = {}

    usage_matrix = mean(data.get_usage_matrix())
    dist_mat = load_data('./distance_matrices/fc_ii_ua_based_sem_hybrid.csv')
    results.update({'hybrid_fc_based_sem_ua': []})

    vmin = np.amin(dist_mat)
    vmax = np.amax(dist_mat)
    step = (vmax - vmin)/10

    t = vmin
    while(t < vmax):
        mae, rmse, prec, rec = fc_user_user(usage_matrix, dist_mat, '/home/imad/Desktop/PFE/db/ml-100k/ua.test', threshold=t)
        print(mae, rmse, prec, rec)
        results['hybrid_fc_based_sem_ua'].append({'threshold': t, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        t = t + step

    usage_matrix = svd_uu_100k(data.get_usage_matrix())
    dist_mat = load_data('./distance_matrices/fc_ii_svd_based_sem_hybrid.csv')
    results.update({'hybrid_fc_based_sem_svd': []})

    vmin = np.amin(dist_mat)
    vmax = np.amax(dist_mat)
    step = (vmax - vmin)/10

    t = vmin
    while(t < vmax):
        mae, rmse, prec, rec = fc_user_user(usage_matrix, dist_mat, '/home/imad/Desktop/PFE/db/ml-100k/ua.test', threshold=t)
        print(mae, rmse, prec, rec)
        results['hybrid_fc_based_sem_svd'].append({'threshold': t, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        t = t + step

    json.dump(results, open('./results/hyb_fc_based_sem.json', 'w'))

    results = {}

    usage_matrix = mean(data.get_usage_matrix())
    dist_mat = load_data('./distance_matrices/sem_based_fc_ua_dist_mat.csv')
    results.update({'hybrid_sem_based_fc_ua': []})

    vmin = np.amin(dist_mat)
    vmax = np.amax(dist_mat)
    step = (vmax - vmin)/10

    t = vmin
    while(t < vmax):
        mae, rmse, prec, rec = fc_user_user(usage_matrix, dist_mat, '/home/imad/Desktop/PFE/db/ml-100k/ua.test', threshold=t)
        print(mae, rmse, prec, rec)
        results['hybrid_sem_based_fc_ua'].append({'threshold': t, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        t = t + step

    usage_matrix = svd_uu_100k(data.get_usage_matrix())
    dist_mat = load_data('./distance_matrices/sem_based_fc_svd_dist_mat.csv')
    results.update({'hybrid_sem_based_fc_svd': []})

    vmin = np.amin(dist_mat)
    vmax = np.amax(dist_mat)
    step = (vmax - vmin)/10

    t = vmin
    while(t < vmax):
        mae, rmse, prec, rec = fc_user_user(usage_matrix, dist_mat, '/home/imad/Desktop/PFE/db/ml-100k/ua.test', threshold=t)
        print(mae, rmse, prec, rec)
        results['hybrid_sem_based_fc_svd'].append({'threshold': t, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        t = t + step

    json.dump(results, open('./results/hyb_sem_based_fc.json', 'w'))

def run_test_knn(data):
    results = {}
    '''
    results.update({'hybrid_alpha_uu_ua': []})
    usage_matrix = mean(data.get_usage_matrix())
    
    test_dict = createDictTestMovies('/home/imad/Desktop/PFE/db/ml-100k/ua.test')

    #fc_dist_mat = load_data('./distance_matrices/fc_ii_ia_dist_mat.csv')
    #fc_ua_dist_mat = load_data('./distance_matrices/fc_ii_ua_dist_mat.csv')
    #fc_svd_dist_mat = load_data('./distance_matrices/fc_ii_svd_dist_mat.csv')
    
    
    #sem_dist_mat = load_data('./distance_matrices/sem_uu_ua_dist_mat.csv')
    fc_dist_mat = load_data('./distance_matrices/fc_uu_ua_dist_mat.csv')
    sem_dist_mat = load_data('./distance_matrices/sem_uu_ua_dist_mat.csv')

    #dist_mat, mae, rmse = best_alpha_beta_hybrid(fc_dist_mat, sem_dist_mat, lambda x: hybrid_knn_eval(152, x, usage_matrix, test_dict))
    
    for k in range(10, 700, 5):
        dist_mat, mae, rmse, prec, rec, alpha = best_alpha_beta_hybrid(
            fc_dist_mat,
            sem_dist_mat,
            lambda x: hybrid_knn_eval(k, x, usage_matrix, test_dict)
        )
        results['hybrid_alpha_uu_ua'].append({'k': k, 'alpha': alpha, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})

    results.update({'hybrid_alpha_uu_svd': []})
    usage_matrix = svd_uu_100k(data.get_usage_matrix())

    #test_dict = createDictTestMovies('/home/imad/Desktop/PFE/db/ml-100k/ua.test')

    #fc_dist_mat = load_data('./distance_matrices/fc_ii_ia_dist_mat.csv')
    #fc_ua_dist_mat = load_data('./distance_matrices/fc_ii_ua_dist_mat.csv')
    #fc_svd_dist_mat = load_data('./distance_matrices/fc_ii_svd_dist_mat.csv')
    
    
    #sem_dist_mat = load_data('./distance_matrices/sem_uu_ua_dist_mat.csv')
    fc_dist_mat = load_data('./distance_matrices/fc_uu_svd_dist_mat.csv')
    sem_dist_mat = load_data('./distance_matrices/sem_uu_svd_dist_mat.csv')

    #dist_mat, mae, rmse = best_alpha_beta_hybrid(fc_dist_mat, sem_dist_mat, lambda x: hybrid_knn_eval(152, x, usage_matrix, test_dict))
    for k in range(10, 700, 5):
        dist_mat, mae, rmse, prec, rec, alpha = best_alpha_beta_hybrid(
            fc_dist_mat,
            sem_dist_mat,
            lambda x: hybrid_knn_eval(k, x, usage_matrix, test_dict)
        )
        results['hybrid_alpha_uu_svd'].append({'k': k, 'alpha': alpha, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})

    json.dump(results, open('./results/hyb_alpha_knn.json', 'w'))

    results = {}

    usage_matrix = mean(data.get_usage_matrix())
    dist_mat = load_data('./distance_matrices/fc_ii_ua_based_sem_hybrid.csv')
    results.update({'hybrid_fc_based_sem_ua': []})

    for k in range(10, 700, 5):
        mae, rmse, prec, rec = hybrid_knn_eval(k, dist_mat, usage_matrix, test_dict)
        print(mae, rmse, prec, rec)
        results['hybrid_fc_based_sem_ua'].append({'k': k, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})

    usage_matrix = svd_uu_100k(data.get_usage_matrix())
    dist_mat = load_data('./distance_matrices/fc_ii_svd_based_sem_hybrid.csv')
    results.update({'hybrid_fc_based_sem_svd': []})

    for k in range(10, 700, 5):
        mae, rmse, prec, rec = hybrid_knn_eval(k, dist_mat, usage_matrix, test_dict)
        print(mae, rmse, prec, rec)
        results['hybrid_fc_based_sem_svd'].append({'k': k, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})

    json.dump(results, open('./results/hyb_fc_based_sem_knn.json', 'w'))

    results = {}
    
    usage_matrix = mean(data.get_usage_matrix())
    dist_mat = load_data('./distance_matrices/sem_based_fc_ua_dist_mat.csv')
    results.update({'hybrid_sem_based_fc_ua': []})

    for k in range(10, 700, 5):
        mae, rmse, prec, rec = hybrid_knn_eval(k, dist_mat, usage_matrix, test_dict)
        print(mae, rmse, prec, rec)
        results['hybrid_sem_based_fc_ua'].append({'k': k, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
    '''
    usage_matrix = svd_uu_100k(data.get_usage_matrix(), './svd_ml_100k.sav')
    dist_mat = load_data('./distance_matrices/sem_based_fc_2_svd_dist_mat.csv')
    results.update({'hybrid_sem_based_fc_svd': []})
    test_dict = createDictTestMovies('/home/imad/Desktop/PFE/db/ml-100k/ua.test')
    for k in range(10, 700, 5):
        mae, rmse, prec, rec = hybrid_knn_eval(k, dist_mat, usage_matrix, test_dict)
        print(mae, rmse, prec, rec)
        results['hybrid_sem_based_fc_svd'].append({'k': k, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        json.dump(results, open('./results/hyb_sem_based_fc_2_knn.json', 'w'))


def run_test_kmedoids(data):
    '''
    results = {}
    
    results.update({'hybrid_alpha_uu_ua': []})
    usage_matrix = mean(data.get_usage_matrix())
    users = [i for i in range(len(usage_matrix))]
    #test_dict = createDictTestMovies('/home/imad/Desktop/PFE/db/ml-100k/ua.test')

    #fc_dist_mat = load_data('./distance_matrices/fc_ii_ia_dist_mat.csv')
    #fc_ua_dist_mat = load_data('./distance_matrices/fc_ii_ua_dist_mat.csv')
    #fc_svd_dist_mat = load_data('./distance_matrices/fc_ii_svd_dist_mat.csv')
    
    
    #sem_dist_mat = load_data('./distance_matrices/sem_uu_ua_dist_mat.csv')
    fc_dist_mat = load_data('./distance_matrices/fc_uu_ua_dist_mat.csv')
    sem_dist_mat = load_data('./distance_matrices/sem_uu_ua_dist_mat.csv')

    #dist_mat, mae, rmse = best_alpha_beta_hybrid(fc_dist_mat, sem_dist_mat, lambda x: hybrid_knn_eval(152, x, usage_matrix, test_dict))
    
    for med_count in range(10, 700, 5):
        initial_medoids = np.random.choice(
            users,
            med_count
        )
        #meds = [int(x) for x in initial_medoids]
        dist_mat, mae, rmse, prec, rec, alpha = best_alpha_beta_hybrid(
            fc_dist_mat,
            sem_dist_mat,
            lambda x: hybrid_kmedoids_eval(initial_medoids, x, usage_matrix, '/home/imad/Desktop/PFE/db/ml-100k/ua.test')
        )
        #print(meds)
        print(mae, rmse)
        results['hybrid_alpha_uu_ua'].append({'meds_count': med_count, 'alpha': alpha, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})

    results.update({'hybrid_alpha_uu_svd': []})
    usage_matrix = svd_uu_100k(data.get_usage_matrix())

    #test_dict = createDictTestMovies('/home/imad/Desktop/PFE/db/ml-100k/ua.test')

    #fc_dist_mat = load_data('./distance_matrices/fc_ii_ia_dist_mat.csv')
    #fc_ua_dist_mat = load_data('./distance_matrices/fc_ii_ua_dist_mat.csv')
    #fc_svd_dist_mat = load_data('./distance_matrices/fc_ii_svd_dist_mat.csv')
    
    
    #sem_dist_mat = load_data('./distance_matrices/sem_uu_ua_dist_mat.csv')
    fc_dist_mat = load_data('./distance_matrices/fc_uu_svd_dist_mat.csv')
    sem_dist_mat = load_data('./distance_matrices/sem_uu_svd_dist_mat.csv')

    #dist_mat, mae, rmse = best_alpha_beta_hybrid(fc_dist_mat, sem_dist_mat, lambda x: hybrid_knn_eval(152, x, usage_matrix, test_dict))
    for med_count in range(10, 700, 5):
        initial_medoids = np.random.choice(
            users,
            med_count
        )
        #meds = [int(x) for x in initial_medoids]
        dist_mat, mae, rmse, prec, rec, alpha = best_alpha_beta_hybrid(
            fc_dist_mat,
            sem_dist_mat,
            lambda x: hybrid_kmedoids_eval(initial_medoids, x, usage_matrix, '/home/imad/Desktop/PFE/db/ml-100k/ua.test')
        )
        #print(meds)
        print(mae, rmse)
        results['hybrid_alpha_uu_svd'].append({'meds_count': med_count, 'alpha': alpha, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})

    json.dump(results, open('./results/hyb_alpha_kmedoids.json', 'w'))

    '''
    results = {}
    '''
    usage_matrix = mean(data.get_usage_matrix())
    dist_mat = load_data('./distance_matrices/fc_ii_ua_based_sem_hybrid.csv')
    results.update({'hybrid_fc_based_sem_ua': []})

    for med_count in range(10, 700, 5):
        initial_medoids = np.random.choice(
            users,
            med_count
        )
        #meds = [int(x) for x in initial_medoids]
        mae, rmse, prec, rec = hybrid_kmedoids_eval(initial_medoids, dist_mat, usage_matrix, '/home/imad/Desktop/PFE/db/ml-100k/ua.test')
        #print(meds)
        print(mae, rmse)
        results['hybrid_fc_based_sem_ua'].append({'meds_count': med_count, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})
    '''
    usage_matrix = svd_uu_100k(data.get_usage_matrix(), './svd_ml_100k.sav')
    dist_mat1 = load_data('./distance_matrices/fc_uu_svd_dist_mat.csv')
    dist_mat2 = load_data('./distance_matrices/sem_uu_svd_dist_mat.csv')
    results.update({'hybrid_fc_based_sem_svd': []})
    test_dict = createDictTestMovies('/home/imad/Desktop/PFE/db/ml-100k/ua.test')
    for med_count in range(10, 101, 5):
        initial_medoids = np.random.choice(
            len(usage_matrix),
            med_count
        )
        #meds = [int(x) for x in initial_medoids]
        (mae, rmse, prec, rec), meds = hybrid_kmedoids_eval(initial_medoids, dist_mat, usage_matrix, test_dict)
        dist_mat, _, _, _, _ = best_alpha_beta_hybrid(dist_mat1, dist_mat2, lambda x: hybrid_kmedoids_eval(meds, x, usage_matrix, test_dict))
        #print(meds)
        bso = BSO(
            len(usage_matrix),
            meds,
            lambda x: bso_eval(x, dist_mat, usage_matrix, test_dict),
            from_count=False
        )
        (mae, rmse, prec, rec), meds = bso.run(7, global_max_iter=10)
        print(med_count)
        print(mae, rmse, prec, rec)
        results['hybrid_fc_based_sem_svd'].append({'meds_count': med_count, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})
        json.dump(results, open('./results/hyb_fc_based_sem_kmedoids_bso_10.json', 'w'))

    '''
    results = {}

    usage_matrix = mean(data.get_usage_matrix())
    dist_mat = load_data('./distance_matrices/sem_based_fc_ua_dist_mat.csv')
    results.update({'hybrid_sem_based_fc_ua': []})

    for med_count in range(10, 700, 5):
        initial_medoids = np.random.choice(
            users,
            med_count
        )
        #meds = [int(x) for x in initial_medoids]
        mae, rmse, prec, rec = hybrid_kmedoids_eval(initial_medoids, dist_mat, usage_matrix, '/home/imad/Desktop/PFE/db/ml-100k/ua.test')
        #print(meds)
        print(mae, rmse)
        results['hybrid_sem_based_fc_ua'].append({'meds_count': med_count, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})
    
    usage_matrix = svd_uu_100k(data.get_usage_matrix())
    dist_mat = load_data('./distance_matrices/sem_based_fc_svd_dist_mat.csv')
    test_dict = createDictTestMovies('/home/imad/Desktop/PFE/db/ml-100k/ua.test')
    results.update({'hybrid_sem_based_fc_svd': []})

    for med_count in range(10, 101, 5):
        initial_medoids = np.random.choice(
            len(usage_matrix),
            med_count
        )
        #meds = [int(x) for x in initial_medoids]
        (mae, rmse, prec, rec), meds = hybrid_kmedoids_eval(initial_medoids, dist_mat, usage_matrix, test_dict)
        #print(meds)
        print(med_count, mae, rmse, prec, rec)
        bso = BSO(len(usage_matrix), meds, lambda x: bso_eval(x, dist_mat, usage_matrix, test_dict), from_count=False)
        (mae, rmse, prec, rec), _ = bso.run(7)
        results['hybrid_sem_based_fc_svd'].append({'meds_count': med_count, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})
        json.dump(results, open('./results/hyb_sem_based_fc_kmedoids_bso.json', 'w'))
    '''

if __name__ == "__main__":
    
    data = DataSet(
        '/home/imad/Desktop/PFE/db/ml-100k/ua.base',
        '/home/imad/Desktop/PFE/db/ml-100k/u.user',
        '/home/imad/Desktop/PFE/db/ml-100k/u.item'
    )
    '''
    test_dict = createDictTestMovies('/home/imad/Desktop/PFE/db/ml-100k/ua.test')
    #run_test_kmedoids(data)
    usage_matrix = svd_uu_100k(data.get_usage_matrix())
    dist_mat = load_data('./distance_matrices/fc_ii_svd_based_sem_hybrid.csv')
    result = {'hybrid_fc_based_sem_svd_knn': []}
    for k in range(20, 40):
        mae, rmse, prec, rec = hybrid_knn_eval(k, dist_mat, usage_matrix, test_dict)
        result['hybrid_fc_based_sem_svd_knn'].append({'k': k, 'mae': mae, 'rmse': rmse, 'prec': prec, 'rec': rec})
        print(k, mae, rmse, prec, rec)
    
    json.dump(result, open('./results/hyb_fc_based_sem_knn_25_40.json', 'w'))
    
    print("hybrid_sem_based_fc_svd | kmedoids | bso")
    run_test_kmedoids(data)

    
    fc_dist = load_data('./distance_matrices/fc_uu_ua_dist_mat.csv')
    sem_dist = load_data('./distance_matrices/sem_uu_ua_dist_mat.csv')

    dist_mat = semantic_based_fc_hybrid(sem_dist, fc_dist)
    dump_array(dist_mat, './distance_matrices/sem_based_fc_ua_dist_mat.csv') 
    
    fc_dist = load_data('./distance_matrices/fc_uu_svd_dist_mat.csv')
    sem_dist = load_data('./distance_matrices/sem_uu_svd_dist_mat.csv')

    dist_mat = semantic_based_fc_hybrid(sem_dist, fc_dist)
    dump_array(dist_mat, './distance_matrices/sem_based_fc_2_svd_dist_mat.csv')
    '''
    run_test_knn(data)
