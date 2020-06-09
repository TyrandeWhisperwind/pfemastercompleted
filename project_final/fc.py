from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from dataset import DataSet, load_data, dump_array
from preprocess import mean, svd_uu_100k, svd_ii_100k, gnb_uu_100k, gnb_ii_100k
import numpy as np
import sklearn.externals.joblib as joblib

def pearson_distance(arr: np.array) -> np.array:
        return 0.5*np.array(
            cosine_distances(
                [
                    [
                        x - np.sum(line)/np.count_nonzero(line) if x != 0
                        else 0
                        for x in line
                    ]
                    for line in arr
                ]
            )
        )


if __name__ == "__main__":
    #a = np.array([[4.0,0.0,0.0,5.0,1.0,0.0,0.0],[5.0,5.0,4.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,2.0,4.0,5.0,0.0],[0.0,3.0,0.0,0.0,0.0,0.0,3.0]])
    '''
    ca = (1 - np.around(cosine_similarity(
        [
            [
                x - np.sum(line)/np.count_nonzero(line) if x != 0
                else 0
                for x in line
            ]
            for line in a
        ]
    ), decimals=6))/2

    #print(1-np.array([[1.,2,3,4,5,6], [10,20,30,40,50,60]]))
    
    ca = np.around(cosine_distances(
        [
            [
                x - np.sum(line)/np.count_nonzero(line) if x != 0
                else 0
                for x in line
            ]
            for line in a
        ]
    ), decimals=6)
    print(ca)


    
    data = DataSet(
        '/home/imad/Desktop/PFE/db/ml-100k/ua.base',
        '/home/imad/Desktop/PFE/db/ml-100k/u.user',
        '/home/imad/Desktop/PFE/db/ml-100k/u.item'
    )
    
    ##FC user-user
    #fc_uu_dist_mat = pearson_distance(data.get_usage_matrix())
    #joblib.dump(fc_uu_dist_mat, 'distance_matrices/fc_uu_dist_mat.sav')
    #dump_array(fc_uu_dist_mat, 'distance_matrices/fc_uu_dist_mat.csv')

    ##FC user-user user average
    fc_uu_ua_dist_mat = pearson_distance(mean(data.get_usage_matrix(), axis=1))
    #joblib.dump(fc_uu_ua_dist_mat, 'distance_matrices/fc_uu_ua_dist_mat.sav')
    dump_array(fc_uu_ua_dist_mat, 'distance_matrices/fc_uu_ua_dist_mat.csv')

    ##FC user-user item average
    fc_uu_ia_dist_mat = pearson_distance(mean(data.get_usage_matrix(), axis=0))
    #joblib.dump(fc_uu_ia_dist_mat, 'distance_matrices/fc_uu_ia_dist_mat.sav')
    dump_array(fc_uu_ia_dist_mat, 'distance_matrices/fc_uu_ia_dist_mat.csv')

    ##FC user-user SVD
    fc_uu_svd_dist_mat = pearson_distance(svd_uu_100k(data.get_usage_matrix()))
    #joblib.dump(fc_uu_svd_dist_mat, 'distance_matrices/fc_uu_svd_dist_mat.sav')
    dump_array(fc_uu_svd_dist_mat, 'distance_matrices/fc_uu_svd_dist_mat.csv')

    ##FC user-user GNB
    fc_uu_gnb_dist_mat = pearson_distance(gnb_uu_100k(data.get_usage_matrix()))
    #joblib.dump(fc_uu_svd_dist_mat, 'distance_matrices/fc_uu_svd_dist_mat.sav')
    dump_array(fc_uu_gnb_dist_mat, 'distance_matrices/fc_uu_gnb_dist_mat.csv')

    ###############################################################################
    data.set_usage_matrix(np.transpose(data.get_usage_matrix()))

    ##FC item-item
    fc_ii_dist_mat = pearson_distance(data.get_usage_matrix())
    #joblib.dump(fc_ii_dist_mat, 'distance_matrices/fc_ii_dist_mat.sav')
    dump_array(fc_ii_dist_mat, 'distance_matrices/fc_ii_dist_mat.csv')

    ##FC item-item user average
    fc_ii_ua_dist_mat = pearson_distance(mean(data.get_usage_matrix(), axis=0))
    #joblib.dump(fc_ii_ua_dist_mat, 'distance_matrices/fc_ii_ua_dist_mat.sav')
    dump_array(fc_ii_ua_dist_mat, 'distance_matrices/fc_ii_ua_dist_mat.csv')

    ##FC item-item item average
    fc_ii_ia_dist_mat = pearson_distance(mean(data.get_usage_matrix(), axis=1))
    #joblib.dump(fc_ii_ia_dist_mat, 'distance_matrices/fc_ii_ia_dist_mat.sav')
    dump_array(fc_ii_ia_dist_mat, 'distance_matrices/fc_ii_ia_dist_mat.csv')

    ##FC item-item SVD
    fc_ii_svd_dist_mat = pearson_distance(svd_ii_100k(data.get_usage_matrix()))
    #joblib.dump(fc_ii_svd_dist_mat, 'distance_matrices/fc_ii_svd_dist_mat.sav')
    dump_array(fc_ii_svd_dist_mat, 'distance_matrices/fc_ii_svd_dist_mat.csv')


    ##FC item-item GNB
    fc_ii_gnb_dist_mat = pearson_distance(gnb_ii_100k(data.get_usage_matrix()))
    #joblib.dump(fc_ii_svd_dist_mat, 'distance_matrices/fc_ii_svd_dist_mat.sav')
    dump_array(fc_ii_gnb_dist_mat, 'distance_matrices/fc_ii_gnb_dist_mat.csv')
    '''
