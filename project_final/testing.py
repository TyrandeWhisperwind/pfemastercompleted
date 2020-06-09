from dataset import DataSet, load_data, dump_array
from semantic import semantic, wp_distance, compute_item_matrix, get_parents
from fc import pearson_distance
import pickle
import numpy as np
'''
import fc_user_user
fc_user_user.main()
import fc_item_item
fc_item_item.main()
import fc_knn
import fc_kmedoids
import semantic_user_user
semantic_user_user.main()
import semantic_knn
semantic_knn.main()
import semantic_kmedoids
import hybrid_multiview_kmedoids
'''
#import hybrid_multiview_knn
#import hybrid

#ratings = load_data('../../db/epinions/ea.base', sep='\t')

data = pickle.load(open('../../db/epinions/epinions-svd-new.sav', 'rb'))
usage_matrix = np.transpose(data.get_usage_matrix())

fc_ii_svd_dist_mat = pearson_distance(usage_matrix)
dump_array(fc_ii_svd_dist_mat, 'distance_matrices/epinions/fc_ii_svd_dist_mat.csv')
#print(np.count_nonzero(np.sum(usage_matrix, axis=1)))
#print(get_parents(278, tax))
#item_dist = compute_item_matrix(data.get_movie_matrix(), wp_distance, tax)
#print(item_dist)
#dist_mat = semantic(data.get_usage_matrix(), data.get_movie_matrix(), wp_distance, tax)

#dump_array(dist_mat, './distance_matrices/epinions/sem_wp_uu_svd_dist_mat.csv')
