import pandas as pd
from scipy.spatial import distance
from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math 

#np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})
'''
d = np.array([
    [4,2,0,3],
    [1,0,0,5],
    [0,3,1,0]
])
'''


def meanRatings(usageMatrix):
    listeMeanRatings=[]
    um = np.array(usageMatrix)
    for i in range(len(usageMatrix)):
        rating=np.sum(np.trim_zeros(usageMatrix[i]))/np.count_nonzero(usageMatrix[i])
        um[i][um[i] == 0.] = rating
        listeMeanRatings.append(rating)
    #return um
    return (um, listeMeanRatings)

#x, y = meanRatings(d)
#print(x)
#print(y)

#########################################################################
#read a file and create matrice user user with cosin similarity
def creatMatrice(usagearrays):
    for i in range(len(usagearrays)):
        avg=sum(usagearrays[i])/np.count_nonzero(usagearrays[i])
        for j in range(len(usagearrays[i])):
            if(usagearrays[i,j]!=0):
                usagearrays[i,j]=usagearrays[i,j]-avg        
    matrice = cosine_similarity(usagearrays) #if similarity we use max if distance we use min 
    matrice=(1.-matrice)/2.
    np.fill_diagonal(matrice, np.nan)#not taking the element itself when counting
    return matrice
#################################################################################################
#read a test file and create a dictionary of movies that we need to guess
def createDictTestMovies(testFile):
    # test file 
    movieDict =  defaultdict(list)#list of a couple 
    with open(testFile, mode='r', encoding='UTF-8') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            userID = int(fields[0])-1 #users are from 0 to 942
            movieID = int(fields[1])-1
            rating = fields[2]
            movieDict[userID].append((movieID,rating))
    return movieDict
#################################################################################################
def getNeighbours(k,matrice):
    userNeighbours=defaultdict(list)
    for x in range(len(matrice)):#0~942 users
        ligne=matrice[x]
        for cpt in range(k):
            userNeighbours[x].append(np.nanargmin(ligne))#if similarity we use max if distance we use min 
            ligne[np.nanargmin(ligne)]=np.nan #remove max element from the list to get the next max element 
    return userNeighbours
#################################################################################################
def knn(k,baseFile,testFile):
    ratings = pd.read_csv(baseFile,sep='\t',names=['user','movie','rating','time'])
    usagematrix = ratings.pivot_table(index='user', columns='movie', values='rating').fillna(0) 
    usagearrays=usagematrix.values

    #print("matrice d'usage")
    #print(usagearrays)
    listeMeanRatings=meanRatings(usagearrays)
    userNeighbours= getNeighbours(k,creatMatrice(usagearrays))
    movieDict=createDictTestMovies(testFile)
  
    aze= ratings.pivot_table(index='user', columns='movie', values='rating').fillna(0) 
    aze=aze.values
    realValue=[]
    predection=[]
    for x in range(len(aze)):#0~942 users ids
        #get the id of movies in test file of user x
            listOfMovies=movieDict[x]#0~942 users ids: get the movies of user x
            for (val, element) in listOfMovies:
                
                #print("movie",int(val)-1)
                realValue.append(element)#element[val] is rating and int(val)-1 is the id of the movie
                rating=0
                #print("neighbours of the user",x,"=",userNeighbours[x])
                for j in userNeighbours[x]:#get the neighbours of  user x
                    #test if rating is zero 
                        if (aze[j][int(val)-1]==0.):
                            rating=rating+listeMeanRatings[x]
                        else:
                            rating=rating+aze[j][int(val)-1]#remove 1 cuz in matrix movies are from 0 to ...
                rating=rating/k#average rating of neighobrs for a given movie 
                if rating >5.: 
                    rating=5.
                predection.append(round(rating))

    realValue=list(np.float_(realValue))
    mae=mean_absolute_error(realValue,predection)
    rmse=math.sqrt(mean_squared_error(realValue,predection))
    resutls=[]
    resutls.append(mae)
    resutls.append(rmse)
    #print("real values:",realValue)
    #print("predection:",predection)
    #print("-----------------------")
    print("mean_absolute_error and mean_squared_error=",resutls)
    return resutls
#################################################################################################
#knn(60,"../ua.base","../ua.test")


class KNN:
    def __init__(self, k: int, user_distance, usage_matrix):
        self.__user_dist_matrix = user_distance
        self.__usagematrix = usage_matrix
        self.__k = k


    def createDictTestMovies(self, testFile):
        # test file 
        movieDict =  defaultdict(list)#list of a couple 
        with open(testFile, mode='r', encoding='UTF-8') as f:
            for line in f:
                fields = line.rstrip('\n').split('\t')
                movieDict[int(fields[0])-1].append((int(fields[1])-1,fields[2]))
        return movieDict

    def getNeighbours(self):
        userNeighbours=defaultdict(list)
        for x in range(len(self.__user_dist_matrix)):#0~942 users
            #ligne=self.__user_dist_matrix[x]
            userNeighbours[x] = np.argsort(self.__user_dist_matrix[x])[:self.__k]
            #for _ in range(self.__k):
            #    userNeighbours[x].append(np.nanargmin(ligne))#if similarity we use max if distance we use min 
            #    ligne[userNeighbours[x][-1]]=np.nan #remove max element from the list to get the next max element 
        return userNeighbours
    
    def sommeSim(self, idUser,userNeighbours,distanceMatrice):
        similaritySum=0
        for element in userNeighbours:
            similaritySum+=distanceMatrice[idUser][element]

        return similaritySum

    def getRating(self, neighbours,idMovie,idUser,matrice,distance_matrice, mean_ratings):
        rating=0

        neighbours = list(filter(lambda x: x!=idUser, neighbours))
        if len(neighbours) == 0: return mean_ratings[idUser]
        for cpt in neighbours:
            rating+=distance_matrice[idUser][cpt]*(matrice[cpt][idMovie]-mean_ratings[cpt])
        #remove rating of the user i'm predecting (didn't want to test if cpt!=idUser...)
        
        rating=rating/self.sommeSim(idUser,neighbours,distance_matrice)
        rating=rating+mean_ratings[idUser]
        return rating
    
    def meanRatings(self, usageMatrix):
        listeMeanRatings=[]
        for i in range(len(usageMatrix)):
            rating=np.sum(np.trim_zeros(usageMatrix[i]))/np.count_nonzero(usageMatrix[i])
            listeMeanRatings.append(rating)
        #print(listeMeanRatings)
        return listeMeanRatings
    
    def process(self, movie_dict):
        self.__movieDict = movie_dict
        userNeighbours = self.getNeighbours()
        list_mean_ratings = self.meanRatings(self.__usagematrix)
        realValue=[]
        predection=[]
        for x in range(len(self.__usagematrix)):#0~942 users ids
            #get the id of movies in test file of user x
                listOfMovies=self.__movieDict[x]#0~942 users ids: get the movies of user x
                for (val, element) in listOfMovies:
                    
                    #print("movie",int(val)-1)
                    realValue.append(element)#element[val] is rating and int(val)-1 is the id of the movie
                    rating=0
                    #print("neighbours of the user",x,"=",userNeighbours[x])
                    #rating = np.average(self.__usagematrix[userNeighbours[x]][:, int(val)-1])
                    rating = self.getRating(userNeighbours[x], val, x, self.__usagematrix, self.__user_dist_matrix, list_mean_ratings)
                    #for j in userNeighbours[x]:#get the neighbours of  user x
                        #print(listeMeanRatings[x], self.__usagematrix[j][int(val)-1])
                        #test if rating is zero 
                        #if (self.__usagematrix[j][int(val)-1]==0.):
                        #    rating=rating+listeMeanRatings[j]
                        #else:
                        #    rating=rating+self.__usagematrix[j][int(val)-1]#remove 1 cuz in matrix movies are from 0 to ...
                        #rating=rating+self.__usagematrix[j][int(val)-1]
                    #rating=rating/self.__k#average rating of neighobrs for a given movie 
                    
                    if rating >5.: 
                        rating=5.
                    predection.append(round(rating))
        
        realValue=list(np.float_(realValue))
        self.__mae=mean_absolute_error(realValue,predection)
        self.__rmse=math.sqrt(mean_squared_error(realValue,predection))
        self.__precision = precision_score(realValue, predection, average='micro')
        self.__recall = recall_score(realValue, predection, average='micro')
    
    def mae_rmse(self):
        return (self.__mae, self.__rmse, self.__precision, self.__recall)


'''
ratings = pd.read_csv('../ua.base',sep='\t',names=['user','movie','rating','time'])
usagematrix = ratings.pivot_table(index='user', columns='movie', values='rating').fillna(0) 
mean_um, usagearrays=meanRatings(usagematrix.values)
mum = np.array(mean_um)
#mean_um = meanRatings(usagematrix.values)
matrice = creatMatrice(mean_um)

algo = KNN(90, matrice, usagematrix.values)
print('start KNN')
algo.process(createDictTestMovies("../ua.test"), usagearrays)
print(algo.mae_rmse())
'''

class KNNMultiview:
    def __init__(self, k: int, user_distance1, user_distance2, usage_matrix):
        self.__user_dist_matrix1 = user_distance1
        self.__user_dist_matrix2 = user_distance2
        self.__usagematrix = usage_matrix
        self.__k = k


    def createDictTestMovies(self, testFile):
        # test file 
        movieDict =  defaultdict(list)#list of a couple 
        with open(testFile, mode='r', encoding='UTF-8') as f:
            for line in f:
                fields = line.rstrip('\n').split('\t')
                movieDict[int(fields[0])-1].append((int(fields[1])-1,fields[2]))
        return movieDict

    def getNeighbours(self, user_dist_matrix1, user_dist_matrix2):
        userNeighbours=defaultdict(list)
        n = len(user_dist_matrix1)
        for x in range(n):#0~942 users
            #ligne=self.__user_dist_matrix[x]

            userNeighbours1 = np.argsort(user_dist_matrix1[x])[:self.__k]
            userNeighbours1 = set((i, user_dist_matrix1[x][i]) for i in userNeighbours1)
            #print(userNeighbours1)
            userNeighbours2 = np.argsort(user_dist_matrix2[x])[:self.__k]
            userNeighbours2 = set((i, user_dist_matrix2[x][i]) for i in userNeighbours2)
            #print(userNeighbours2)

            userNeighbours[x] = list(
                sorted(list(userNeighbours1 | userNeighbours2), key=lambda elt: elt[1])[:self.__k]
            )
            #print(userNeighbours[x])
            #print(len(userNeighbours[x]))
            #userNeighbours[x] = np.argsort(list(set(userNeighbours1) | set(userNeighbours2)))
            #for _ in range(self.__k):
            #    userNeighbours[x].append(np.nanargmin(ligne))#if similarity we use max if distance we use min 
            #    ligne[userNeighbours[x][-1]]=np.nan #remove max element from the list to get the next max element 
        return userNeighbours
    
    def sommeSim(self,userNeighbours):
        similaritySum=0
        for _, dist in userNeighbours:
            similaritySum+=dist

        return similaritySum

    def getRating(self, neighbours,idMovie,idUser,matrice, mean_ratings):
        rating=0

        neighbours = list(filter(lambda x: x[0]!=idUser, neighbours))
        if len(neighbours) == 0: return mean_ratings[idUser]
        for cpt, dist in neighbours:
            rating+=dist*(matrice[cpt][idMovie]-mean_ratings[cpt])
        #remove rating of the user i'm predecting (didn't want to test if cpt!=idUser...)
        
        rating=rating/self.sommeSim(neighbours)
        rating=rating+mean_ratings[idUser]
        return rating
    
    def meanRatings(self, usageMatrix):
        listeMeanRatings=[]
        for i in range(len(usageMatrix)):
            rating=np.sum(np.trim_zeros(usageMatrix[i]))/np.count_nonzero(usageMatrix[i])
            listeMeanRatings.append(rating)
        #print(listeMeanRatings)
        return listeMeanRatings
    
    def process(self, movie_dict):
        self.__movieDict = movie_dict
        userNeighbours = self.getNeighbours(self.__user_dist_matrix1, self.__user_dist_matrix2)
        list_mean_ratings = self.meanRatings(self.__usagematrix)
        realValue=[]
        predection=[]
        for x in range(len(self.__usagematrix)):#0~942 users ids
            #get the id of movies in test file of user x
                listOfMovies=self.__movieDict[x]#0~942 users ids: get the movies of user x
                for (val, element) in listOfMovies:
                    
                    #print("movie",int(val)-1)
                    realValue.append(element)#element[val] is rating and int(val)-1 is the id of the movie
                    rating=0
                    #print("neighbours of the user",x,"=",userNeighbours[x])
                    #rating = np.average(self.__usagematrix[userNeighbours[x]][:, int(val)-1])
                    rating = self.getRating(userNeighbours[x], val, x, self.__usagematrix, list_mean_ratings)
                    #for j in userNeighbours[x]:#get the neighbours of  user x
                        #print(listeMeanRatings[x], self.__usagematrix[j][int(val)-1])
                        #test if rating is zero 
                        #if (self.__usagematrix[j][int(val)-1]==0.):
                        #    rating=rating+listeMeanRatings[j]
                        #else:
                        #    rating=rating+self.__usagematrix[j][int(val)-1]#remove 1 cuz in matrix movies are from 0 to ...
                        #rating=rating+self.__usagematrix[j][int(val)-1]
                    #rating=rating/self.__k#average rating of neighobrs for a given movie 
                    
                    if rating >5.: 
                        rating=5.
                    predection.append(round(rating))
        
        realValue=list(np.float_(realValue))
        self.__mae=mean_absolute_error(realValue,predection)
        self.__rmse=math.sqrt(mean_squared_error(realValue,predection))
        realValue = [round(v) for v in realValue]
        predection = [round(v) for v in predection]
        self.__precision = precision_score(realValue, predection, average='macro')
        self.__recall = recall_score(realValue, predection, average='macro')
        print("mean_absolute_error and mean_squared_error=",(self.__mae,self.__rmse,self.__precision,self.__recall))
    
    def mae_rmse(self):
        return (self.__mae, self.__rmse, self.__precision, self.__recall)
