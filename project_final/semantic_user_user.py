import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import mean, svd_uu_100k, gnb_uu_100k
from collections import defaultdict
from dataset import DataSet, load_data
from semantic import modified_semantic
import numpy as np
import json
#import sklearn.externals.joblib as joblib
#from fc import pearson_distance

def sommeSim(idUser,userNeighbours,distanceMatrice):
    similaritySum=0
    for element in userNeighbours:
        similaritySum+=distanceMatrice[idUser][element]
    return similaritySum

def meanRatings(usageMatrix):
    listeMeanRatings=[]
    for i in range(len(usageMatrix)):
        rating=np.sum(np.trim_zeros(usageMatrix[i]))/np.count_nonzero(usageMatrix[i])
        listeMeanRatings.append(rating)
    return listeMeanRatings

def getNeighboursSeuil(seuil,matrice):
    userNeighbours=defaultdict(list)
    for x in range(len(matrice)):#0~942 users
        ligne=matrice[x]
        for cpt in range(len(ligne)):
            if ((ligne[cpt]<seuil) and (cpt!=x)):
                userNeighbours[x].append(cpt)
    return userNeighbours

def createDictTestMovies(testFile):
    # test file 
    movieDict =  defaultdict(list)#list of a couple 
    with open(testFile, mode='r', encoding='UTF-8') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            userID = int(fields[0])-1 #users are from 0 to 942
            movieID = fields[1]
            rating = fields[2]
            movieDict[userID].append({ movieID:rating })
    return movieDict

def predictionAndError(matriceUsage,testFile,listeofneighbours,listemeanRAting,distanceMatrice):
    movieDict=createDictTestMovies(testFile)
    realValue=[]
    predection=[]
    for x in range(len(matriceUsage)):#0~942 users ids
        #get the id of movies in test file of user x
            listOfMovies=movieDict[x]#0~942 users ids: get the movies of user x
            for element in listOfMovies:
                for val in element:
                    #print("movie",int(val)-1)
                    realValue.append(element[val])#element[val] is rating and int(val)-1 is the id of the movie
                    numerateur=0
                    rating=0
                    if len(listeofneighbours[x])==0:#si pas de voisin alors affecté la moyenne des rating de x
                        rating=listemeanRAting[x]
                    else:
                        for user in listeofneighbours[x]:
                            numerateur+=distanceMatrice[x][user]*(matriceUsage[user][int(val)-1]-listemeanRAting[user])
                        
                        numerateur=numerateur/sommeSim(x,listeofneighbours[x],distanceMatrice)
                        rating=numerateur+listemeanRAting[x]

                    predection.append(round(rating))
    realValue=list(np.float_(realValue))
    mae=mean_absolute_error(realValue,predection)
    rmse=np.sqrt(mean_squared_error(realValue,predection))
    prec = precision_score(realValue, predection, average='micro')
    rec = recall_score(realValue, predection, average='micro')

    print("mean_absolute_error and mean_squared_error=",(mae, rmse, prec, rec))
    return (mae, rmse, prec, rec)


def semantic_user_user(usage_matrix, distance_matrix, test_file, threshold=0.43):
    neighbours_dict = getNeighboursSeuil(threshold, distance_matrix)
    list_mean_ratings = meanRatings(usage_matrix)
    return predictionAndError(usage_matrix, test_file, neighbours_dict, list_mean_ratings, distance_matrix)


def main():
    
    data = DataSet(
    '/home/imad/Desktop/PFE/db/ml-100k/ua.base',
    '/home/imad/Desktop/PFE/db/ml-100k/u.user',
    '/home/imad/Desktop/PFE/db/ml-100k/u.item'
    )
    
    results = {}

    print("#Semantic user user")
    results.update({'sem_user_user': []})
    usage_matrix = data.get_usage_matrix()
    distance_matrix = load_data('./distance_matrices/sem_uu_dist_mat.csv')
    
    vmin = np.amin(distance_matrix)
    vmax = np.amax(distance_matrix)
    step = (vmax - vmin)/10

    t = vmin
    while t <= vmax:
        mae, rmse, pres, rec = semantic_user_user(usage_matrix, distance_matrix, "/home/imad/Desktop/PFE/db/ml-100k/ua.test", threshold=t)
        results['sem_user_user'].append({'threshold': t, 'mae': mae, 'rmse': rmse, 'pres': pres, 'rec': rec})
        t = t + step
    
    print("#Semantic user user | mean user")
    results.update({'sem_user_user_ua': []})
    usage_matrix = mean(data.get_usage_matrix())
    distance_matrix = load_data('./distance_matrices/sem_uu_ua_dist_mat.csv')
    
    vmin = np.amin(distance_matrix)
    vmax = np.amax(distance_matrix)
    step = (vmax - vmin)/10

    t = vmin
    while t <= vmax:
        mae, rmse, pres, rec = semantic_user_user(usage_matrix, distance_matrix, "/home/imad/Desktop/PFE/db/ml-100k/ua.test", threshold=t)
        results['sem_user_user_ua'].append({'threshold': t, 'mae': mae, 'rmse': rmse, 'pres': pres, 'rec': rec})
        t = t + step

    
    print("#Semantic user user | mean item")
    results.update({'sem_user_user_ia': []})
    usage_matrix = mean(data.get_usage_matrix(), axis=0)
    distance_matrix = load_data('./distance_matrices/sem_uu_ia_dist_mat.csv')
    
    vmin = np.amin(distance_matrix)
    vmax = np.amax(distance_matrix)
    step = (vmax - vmin)/10

    t = vmin
    while t <= vmax:
        mae, rmse, pres, rec = semantic_user_user(usage_matrix, distance_matrix, "/home/imad/Desktop/PFE/db/ml-100k/ua.test", threshold=t)
        results['sem_user_user_ua'].append({'threshold': t, 'mae': mae, 'rmse': rmse, 'pres': pres, 'rec': rec})
        t = t + step


    print("#Semantic user user | svd")
    results.update({'sem_user_user_svd': []})
    usage_matrix = svd_uu_100k(data.get_usage_matrix())
    distance_matrix = load_data('./distance_matrices/sem_uu_svd_dist_mat.csv')
    
    vmin = np.amin(distance_matrix)
    vmax = np.amax(distance_matrix)
    step = (vmax - vmin)/10

    t = vmin
    while t <= vmax:
        mae, rmse, pres, rec = semantic_user_user(usage_matrix, distance_matrix, "/home/imad/Desktop/PFE/db/ml-100k/ua.test", threshold=t)
        results['sem_user_user_svd'].append({'threshold': t, 'mae': mae, 'rmse': rmse, 'pres': pres, 'rec': rec})
        t = t + step

    '''
    print("#Semantic user user | GNB")
    usage_matrix = gnb_uu_100k(data.get_usage_matrix())
    distance_matrix = load_data('./distance_matrices/sem_uu_gnb_dist_mat.csv')
    mae, rmse = semantic_user_user(usage_matrix, distance_matrix, "/home/imad/Desktop/PFE/db/ml-100k/ua.test", threshold=t)
    results.update({'semantic_user_user_gnb': {'threshold': t, 'mae': mae, 'rmse': rmse}})
    '''
    json.dump(results, open('./results/semantic_user_user.json', 'w'))
    

if __name__ == "__main__":
    main()