import numpy as np
from dataset import DataSet, dump_array
from preprocess import mean, svd_uu_100k, gnb_uu_100k
#from semantic_user_user import semantic_user_user

def jaccard_similarity(x: list, y: list):
    c11, c01, c10 = 0, 0, 0
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 0: continue
        c11 = c11 + x[i] + y[i] - 1
        c10 = c10 + x[i]
        c01 = c01 + y[i]
    return c11/(c10 + c01 + c11)

def jaccard_distance(x: list, y: list):
    return 1 - jaccard_similarity(x, y)


def get_parents(x, taxonomy):
    i = int(x)
    prarents = [x]
    while(taxonomy[i] != -1):
        prarents.append(int(taxonomy[i]))
        i = int(taxonomy[i])
    return prarents

def wp_similarity(x, y, taxonomy):
    p1 = get_parents(x, taxonomy)
    p2 = get_parents(y, taxonomy)


    for i, p in enumerate(p1):
        if p in p2:
            l1 = len(p1[:i])
            l2 = len(p2[:p2.index(p)])
            dept = len(p1[i:])
            return 2*dept/(l1 + l2 + 2*dept)
    
    return 0

def wp_distance(x, y, taxonomy):
    return 1 - wp_similarity(x, y, taxonomy)

def get_interest_dict(users: np.array) ->dict:
    (N, M) = users.shape
    return {
        u: {
            m: users[u, m]
            for m in range(M)
            if users[u, m] > 0
        }
        for u in range(N)
    }


def get_sorted_item_matrix(item_dist_matrix: np.array) ->np.array:
        return {
            
            line:{
                m2:dist
                for m2, dist in sorted(item_dist_matrix[line], key=lambda x: x[1])
            }
            for line in range(len(item_dist_matrix))
            
        }


def interest_dist(u1: dict, u2: dict, item_dict_matrix):
    s1 = 0; s2 = 0
    for m1 in u1:
        min1 = 1
        
        for key in item_dict_matrix[m1]:
            if key in u2.keys():
                min1 = item_dict_matrix[m1][key]
                break
        
        s1 += u1[m1]*min1
        s2 += u1[m1]
    
    for m2 in u2:
        min2 = 1
        
        for key in item_dict_matrix[m2]:
            if key in u1.keys():
                min2 = item_dict_matrix[m2][key]
                break
        
        s1 += u2[m2]*min2
        s2 += u2[m2]

    return s1/s2

def modified_interest_dist(u1: dict, u2: dict, item_dist_matrix):
    #intersect = (set(u1.keys()) & set(u2.keys()))
    #print(intersect)
    #l = len(intersect)
    s1 = 0; s2 = 0
    for m1 in u1:
        if u2.get(m1) is not None:
            s1 += abs(u1[m1] - u2[m1])/4
            s2 += 1
        else:
            min1 = 1
            for m2 in item_dist_matrix[m1]:
                if u2.get(m2) is not None: 
                    min1 = m2
                    break
            s1 += (abs(u1[m1] - u2[min1])/4 + item_dist_matrix[m1][min1])/2
            s2 += 1
    for m2 in u2:
        if u1.get(m2) is None:
            min2 = 1
            for m1 in item_dist_matrix[m2]:
                if u1.get(m1) is not None:
                    min2 = m1
            s1 += (abs(u2[m2] - u1[min2])/4 + item_dist_matrix[m2][min2])/2
            s2 += 1
    

    #s2 = len(u1) + len(u2)
    #print(s2)

    return s1/s2


#compute distance matrix using (interest distance)
def compute_user_matrix(usage_matrix: np.array, item_dict_matrix):
    
    n = len(usage_matrix)

    user_interest_dict = get_interest_dict(usage_matrix)

    user_dist_matrix = np.zeros((n, n))
    for i in range(n):
        #print("line: ", i)
        for j in range(i, n):
            #print("column: ",j, end='\r')
            user_dist_matrix[i, j] = interest_dist(user_interest_dict[i], user_interest_dict[j], item_dict_matrix)
            user_dist_matrix[j, i] = user_dist_matrix[i, j]
    
    return user_dist_matrix

def compute_modified_user_matrix(usage_matrix: np.array, item_dist_matrix):
    
    n = len(usage_matrix)

    user_interest_dict = get_interest_dict(usage_matrix)

    user_dist_matrix = np.zeros((n, n))
    for i in range(n):
        #print("line: ", i)
        for j in range(i, n):
            #print("column: ",j, end='\r')
            user_dist_matrix[i, j] = modified_interest_dist(user_interest_dict[i], user_interest_dict[j], item_dist_matrix)
            user_dist_matrix[j, i] = user_dist_matrix[i, j]
    
    return user_dist_matrix

def compute_item_matrix(items: np.array, metric, *args):
    n = len(items)
    item_dist_matrix = np.zeros((n, n), dtype=tuple)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                item_dist_matrix[i, j] = (j, 1)
                continue    
            item_dist_matrix[i, j] = (j, metric(items[i], items[j], *args))
    
    return item_dist_matrix

def compute_modified_item_matrix(items: np.array, metric):
    n = len(items)
    item_dist_matrix = np.zeros((n, n), dtype=tuple)
    
    for i in range(n):
        for j in range(n):
            item_dist_matrix[i, j] = metric(items[i], items[j])
    
    return item_dist_matrix

def semantic(usage_matrix, item_matrix, metric, *args):
    item_dist_matrix = compute_item_matrix(item_matrix, metric, *args)
    sorted_item_matrix = get_sorted_item_matrix(item_dist_matrix)
    return compute_user_matrix(usage_matrix, sorted_item_matrix)

def modified_semantic(usage_matrix, item_dist_matrix):
    sorted_item_matrix = get_sorted_item_matrix(item_dist_matrix)
    return compute_modified_user_matrix(usage_matrix, sorted_item_matrix)

class Semantic:

    def __init__(self, users: np.array, items: np.array):
        self.__users = users
        self.__items = items
        self.__metric = self.jaccard_distance

        pass



    def process(self):
        self.compute_item_matrix(self.__items)
        self.get_sorted_item_matrix(self.item_dist_matrix)
        self.compute_user_matrix(self.__users)
    
    #gets a dictionary with {user:{movie: rating,...}} if the movie rating > 0  
    def get_interest_dict(self, users: np.array) ->dict:
        (N, M) = users.shape
        return {
            u: {
                m: users[u, m]
                for m in range(M)
                if users[u, m] > 0
            }
            for u in range(N)
        }

    def jaccard_distance(self, x: list, y: list):
        return 1 - self.jaccard_similarity(x, y)

    def jaccard_similarity(self, x: list, y: list):
        c11, c01, c10 = 0, 0, 0
        for i in range(len(x)):
            if x[i] == 0 and y[i] == 0: continue
            c11 = c11 + x[i] + y[i] - 1
            c10 = c10 + x[i]
            c01 = c01 + y[i]
        return c11/(c10 + c01 + c11)

    #calculates movie distances using jaccard metric
    def compute_item_matrix(self, items: np.array):
        n = len(items)
        self.item_dist_matrix = np.zeros((n, n), dtype=tuple)
        
        for i in range(n):
            for j in range(n):
                self.item_dist_matrix[i, j] = (j, self.__metric(items[i], items[j]))
        
        return self.item_dist_matrix

    #sorts neighbours of each movie by ascending order
    def get_sorted_item_matrix(self, item_dist_matrix: np.array) ->np.array:
        self.sorted_item_matrix = {
            
            line:{
                m2:dist
                for m2, dist in sorted(item_dist_matrix[line], key=lambda x: x[1])
            }
            for line in range(len(item_dist_matrix))
            
        }

    #distance between two users (interest distance)
    def interest_dist(self, u1: dict, u2: dict):
        s1 = 0; s2 = 0
        for m1 in u1:
            min1 = 1
            
            for key in self.sorted_item_matrix[m1]:
                if key in u2.keys():
                    min1 = self.sorted_item_matrix[m1][key]
                    break
            
            s1 += u1[m1]*min1
            s2 += u1[m1]
        
        for m2 in u2:
            min2 = 1
            
            for key in self.sorted_item_matrix[m2]:
                if key in u1.keys():
                    min2 = self.sorted_item_matrix[m2][key]
                    break
            
            s1 += u2[m2]*min2
            s2 += u2[m2]

        return s1/s2
   
    #compute distance matrix using (interest distance)
    def compute_user_matrix(self, usage_matrix: np.array):
        if self.item_dist_matrix is None:
            print("no item similarity matrix found")
            pass

        n = len(usage_matrix)

        user_interest_dict = self.get_interest_dict(usage_matrix)

        self.user_dist_matrix = np.zeros((n, n))
        for i in range(n):
            #print("line: ", i)
            for j in range(i, n):
                #print("column: ",j, end='\r')
                self.user_dist_matrix[i, j] = self.interest_dist(user_interest_dict[i], user_interest_dict[j])
                self.user_dist_matrix[j, i] = self.user_dist_matrix[i, j]
        
        return self.user_dist_matrix


if __name__ == "__main__":
    
    d = DataSet('/home/imad/Desktop/PFE/db/ml-100k/ua.base',
        '/home/imad/Desktop/PFE/db/ml-100k/u.user',
        '/home/imad/Desktop/PFE/db/ml-100k/u.item'
    )
    '''
    s = Semantic(d.get_usage_matrix(), d.get_movie_matrix())
    s.process()
    dump_array(s.user_dist_matrix, "./distance_matrices/sem_uu_dist_mat.csv")

    s = Semantic(mean(d.get_usage_matrix()), d.get_movie_matrix())
    s.process()
    dump_array(s.user_dist_matrix, "./distance_matrices/sem_uu_ua_dist_mat.csv")

    s = Semantic(mean(d.get_usage_matrix(), axis=0), d.get_movie_matrix())
    s.process()
    dump_array(s.user_dist_matrix, "./distance_matrices/sem_uu_ia_dist_mat.csv")

    s = Semantic(svd_uu_100k(d.get_usage_matrix()), d.get_movie_matrix())
    s.process()
    dump_array(s.user_dist_matrix, "./distance_matrices/sem_uu_svd_dist_mat.csv")
    
    s = Semantic(gnb_uu_100k(d.get_usage_matrix()), d.get_movie_matrix())
    s.process()
    dump_array(s.user_dist_matrix, "./distance_matrices/sem_uu_gnb_dist_mat.csv")

    '''
    #avg_interest_dist({'1': 3, '2': 1, '5': 2}, {'1': 3, '4': 1, '5': 2})

    item_mat = compute_item_matrix(d.get_movie_matrix(), jaccard_distance)

    dist_matrix = modified_semantic(d.get_usage_matrix(), item_mat)
    dump_array(dist_matrix, "./distance_matrices/mod_sem_uu_dist_mat.csv")

    usage_matrix = mean(d.get_usage_matrix())
    dist_matrix = modified_semantic(usage_matrix, item_mat)
    dump_array(dist_matrix, "./distance_matrices/mod_sem_uu_ua_dist_mat.csv")
