from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score
from collections import defaultdict
from preprocess import mean, svd_ii_100k, gnb_ii_100k
from dataset import DataSet, load_data
import numpy as np
import json

def sommeSim(idUser,userNeighbours,distanceMatrice):
    similaritySum=0
    for element in userNeighbours:
        similaritySum+=distanceMatrice[idUser][element]
    return similaritySum

def meanRatings(usageMatrix):
    listeMeanRatings=[]
    for i in range(len(usageMatrix)):
        rating = np.sum(np.trim_zeros(usageMatrix[i]))
        if rating > 0: rating = rating/np.count_nonzero(usageMatrix[i])
        listeMeanRatings.append(rating)
    return listeMeanRatings

def createDictTestMovies(testFile):
    # test file 
    movieDict =  defaultdict(list)#list of a couple 
    with open(testFile, mode='r', encoding='UTF-8') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            userID = int(fields[0])-1 #users are from 0 to 942
            movieID = int(fields[1])-1 
            rating = fields[2]
            movieDict[movieID].append({ userID:rating })
    return movieDict

def getNeighboursSeuil(seuil,matrice):
    userNeighbours=defaultdict(list)
    for x in range(len(matrice)):#0~942 users
        ligne=matrice[x]
        for cpt in range(len(ligne)):
            if ((ligne[cpt]<seuil) and (cpt!=x)):
                userNeighbours[x].append(cpt)
    return userNeighbours

def predictionAndError(matriceUsage,testFile,listeofneighbours,listemeanRAting,distanceMatrice):
    userDict=createDictTestMovies(testFile)
    realValue=[]
    predection=[]
    for x in range(len(matriceUsage)):
            listOfuser=userDict[x]
            for element in listOfuser:
                for val in element:
                    realValue.append(element[val])#element[val]= rating, val= id of user
                    numerateur=0
                    rating=0
                    if len(listeofneighbours[x])==0:
                        rating=listemeanRAting[x]
                    else:
                        for movie in listeofneighbours[x]:
                            numerateur+=distanceMatrice[x][movie]*(matriceUsage[movie][val]-listemeanRAting[movie])
                        
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


def fc_item_item(usage_matrix, distance_matrix, test_file, threshold=0.43):
    usage_matrix = np.transpose(usage_matrix)
    neighbours_dict = getNeighboursSeuil(threshold, distance_matrix)
    list_mean_ratings = meanRatings(usage_matrix)
    return predictionAndError(usage_matrix, test_file, neighbours_dict, list_mean_ratings, distance_matrix)

def main():
    data = DataSet(
        '/home/imad/Desktop/PFE/db/ml-100k/ua.base',
        '/home/imad/Desktop/PFE/db/ml-100k/u.user',
        '/home/imad/Desktop/PFE/db/ml-100k/u.item'
    )

    results = dict()
    t = 0.43

    print("#FC item item")
    results.update({'fc_item_item': []})
    usage_matrix = data.get_usage_matrix()
    distance_matrix = load_data('./distance_matrices/fc_ii_dist_mat.csv')
    
    vmin = np.amin(distance_matrix)
    vmax = np.amax(distance_matrix)
    step = (vmax - vmin)/10

    t = vmin
    while t <= vmax:
        mae, rmse, prec, rec = fc_item_item(usage_matrix, distance_matrix, "/home/imad/Desktop/PFE/db/ml-100k/ua.test", threshold=t)
        print(t)
        results['fc_item_item'].append({'threshold': t, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})
        t = t + step

    print("#FC item item | user average")
    results.update({'fc_item_item_ua': []})
    usage_matrix = mean(data.get_usage_matrix(), axis=0)
    distance_matrix = load_data('./distance_matrices/fc_ii_ua_dist_mat.csv')
    
    vmin = np.amin(distance_matrix)
    vmax = np.amax(distance_matrix)
    step = (vmax - vmin)/10

    t = vmin
    while t <= vmax:
        mae, rmse, prec, rec = fc_item_item(usage_matrix, distance_matrix, "/home/imad/Desktop/PFE/db/ml-100k/ua.test", threshold=t)
        print(t)
        results['fc_item_item_ua'].append({'threshold': t, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})
        t = t + step

    print("#FC item item | item average")
    results.update({'fc_item_item_ia':[]})
    usage_matrix = mean(data.get_usage_matrix(), axis=1)
    distance_matrix = load_data('./distance_matrices/fc_ii_ia_dist_mat.csv')
    
    vmin = np.amin(distance_matrix)
    vmax = np.amax(distance_matrix)
    step = (vmax - vmin)/10

    t = vmin
    while t <= vmax:
        mae, rmse, prec, rec = fc_item_item(usage_matrix, distance_matrix, "/home/imad/Desktop/PFE/db/ml-100k/ua.test", threshold=t)
        print(t)
        results['fc_item_item_ia'].append({'threshold': t, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})
        t = t + step

    print("#FC item item | SVD")
    results.update({'fc_item_item_svd': []})
    usage_matrix = svd_ii_100k(data.get_usage_matrix())
    distance_matrix = load_data('./distance_matrices/fc_ii_svd_dist_mat.csv')
    
    vmin = np.amin(distance_matrix)
    vmax = np.amax(distance_matrix)
    step = (vmax - vmin)/10

    t = vmin
    while t <= vmax:
        mae, rmse, prec, rec = fc_item_item(usage_matrix, distance_matrix, "/home/imad/Desktop/PFE/db/ml-100k/ua.test", threshold=t)
        print(t)
        results['fc_item_item_svd'].append({'threshold': t, 'mae': mae, 'rmse': rmse, 'pres': prec, 'rec': rec})
        t = t + step

    '''
    print("#FC item item | GNB")
    usage_matrix = gnb_ii_100k(data.get_usage_matrix())
    distance_matrix = load_data('./distance_matrices/fc_ii_gnb_dist_mat.csv')
    mae, rmse = fc_item_item(usage_matrix, distance_matrix, "/home/imad/Desktop/PFE/db/ml-100k/ua.test")
    results.update({'fc_item_item_gnb': {'threshold': t, 'mae': mae, 'rmse': rmse}})
    '''
    json.dump(results, open('./results/fc_item_item.json', 'w'))

if __name__ == "__main__":
    main()