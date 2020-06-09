import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import recall_score, precision_score
from math import sqrt

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
    #print(listeMeanRatings)
    return listeMeanRatings


##############################################################################
def getRating(clusterI,idMovie,idUser,matrice,distance_matrice, mean_ratings):
    rating=0

    neighbours = list(filter(lambda x: x!=idUser, clusterI))
    if len(neighbours) == 0: return mean_ratings[idUser]
    for cpt in neighbours:
        rating+=distance_matrice[idUser][cpt]*(matrice[cpt][idMovie]-mean_ratings[cpt])
    #remove rating of the user i'm predecting (didn't want to test if cpt!=idUser...)
    ss = sommeSim(idUser,neighbours,distance_matrice)
    if ss == 0: return mean_ratings[idUser]    
    rating=rating/ss
    rating=rating+mean_ratings[idUser]
    return rating
#######################################################
#create a dictionary of movies that we need to guess
def createDictTestMovies(testFile):
    # test file 
    movieDict =  defaultdict(list)#list of a couple 
    with open(testFile, mode='r', encoding='UTF-8') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            userID = int(fields[0])-1 #users are from 0 to 942
            movieID = int(fields[1])-1
            rating = float(fields[2])
            movieDict[userID].append({ movieID:rating })
    return movieDict
############################################################################################
def MAE_RMSE(matrice,distance_matrice,Clusters,testFile):
    #usagematrix = ratings.pivot_table(index='user', columns='movie', values='rating')
    #usagematrix=usagematrix.apply(lambda usagematrix: usagematrix.fillna(usagematrix.mean()), axis=1)
    mean_ratings = meanRatings(matrice)
    realValue=[]
    predection=[]
    movieDict = None
    if isinstance(testFile, str):
        movieDict=createDictTestMovies(testFile)
    elif isinstance(testFile, dict):
        movieDict = testFile
    for label in range(len(Clusters)): 
        for idUser in Clusters[label]:
                listOfMovies=movieDict[idUser]
                for element in listOfMovies:
                    for idMovie in element:
                        #print(idUser+1,idMovie)
                        realValue.append(element[idMovie])#element[val] is rating and int(idMovie)-1 is the id of the movie
                        rating=getRating(Clusters[label],idMovie,idUser,matrice, distance_matrice, mean_ratings)
                        if rating >5.:
                                rating=5.

                        predection.append(round(rating))

    #print(list(filter(lambda x: x not in range(1,6), predection)))
    #print(list(filter(lambda x: x not in range(1,6), realValue)))
    #realValue=list(np.float_(realValue))
    print(predection)
    mae=mean_absolute_error(realValue,predection)
    rmse=sqrt(mean_squared_error(realValue,predection))
    realValue = [round(v) for v in realValue]
    predection = [round(v) for v in predection]
    precision = precision_score(realValue, predection, average='micro', labels=[1, 2, 3, 4, 5])
    recall = recall_score(realValue, predection, average='micro', labels=[1, 2, 3, 4, 5])
    
    #print("real values:",realValue)
    #print("predection:",predection)
    print("mean_absolute_error and mean_squared_error=",(mae,rmse,precision,recall))
    return (mae,rmse,precision,recall)