import pandas as pd
from scipy.spatial import distance
from sklearn.metrics.pairwise import *
import numpy as np
import random

from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.metrics.pairwise import *
import math 

"""
#paper's exemple for second algorithm:
CT=[[1,3],[2],[0,4,5]]
CS=[[0,1],[2],[3,4,5]]

medoidsT=[1,2,4]
medoidsS=medoidsT
"""


def functionTest(cluster,id):
    cpt=0
    listecluster=[]
    for element in cluster:
        if id in element:
            listecluster.append(cpt)
        cpt+=1
    return listecluster
###########################################################
def createDictTestMovies(testFile):
    # test file 
    movieDict =  defaultdict(list)#list of a couple 
    with open(testFile, mode='r', encoding='UTF-8') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            userID = int(fields[0])-1 #users are from 0 to 942
            movieID = int(fields[1])-1
            rating = fields[2]
            movieDict[userID].append({ movieID:rating })
    return movieDict
##############################################################
def sommeSim(idUser,userNeighbours,distanceMatrice):
    similaritySum=0
    for element in userNeighbours:
        similaritySum+=distanceMatrice[idUser][element]
    return similaritySum
##############################################################
def meanRatings(usageMatrix):
    listeMeanRatings=[]
    for i in range(len(usageMatrix)):
        rating=np.sum(np.trim_zeros(usageMatrix[i]))/np.count_nonzero(usageMatrix[i])
        listeMeanRatings.append(rating)
    return listeMeanRatings
##############################################################
def ran(labels,medoids):
    cpt=0
    newMedoids=[]
    for element in labels:
        if len(element)>1:
            nu=random.choice(element)
            while (nu==medoids[cpt]):
                nu=random.choice(element)
                if nu!=medoids[cpt]:
                    break
            newMedoids.append(nu)
        else:
            newMedoids.append(medoids[cpt])
        cpt+=1
    return newMedoids
###########################################################
def makeCluster(KSelected,matrice):
    cost =0
    labels = np.empty((len(KSelected), 0)).tolist()

    cpt=0
    for el in matrice :
        values=[]
        for element in KSelected:
            values.append(el[element])
        minIndex=values.index(min(values))
        labels[minIndex].append(cpt)
        cost+=el[KSelected[minIndex]]
        cpt+=1

    liste=[]
    liste.append(labels)
    liste.append(cost)
    return liste
############################################################
def Union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list 
#############################################################
def creatMatrice(usagearrays):
    for i in range(len(usagearrays)):
        avg=sum(usagearrays[i])/np.count_nonzero(usagearrays[i])
        for j in range(len(usagearrays[i])):
            if(usagearrays[i,j]!=0):
                usagearrays[i,j]=usagearrays[i,j]-avg        
    matrice = cosine_similarity(usagearrays) #if similarity we use max if distance we use min 
    matrice=(1.-matrice)/2.
    np.fill_diagonal(matrice, 0.)
    return matrice
##############################################################
def multiView(medoidsT,matrice1,matrice2,maxIteration,threshold):
    alreadyMed=np.empty((len(medoidsT), 0)).tolist()
    cpt=0
    for element in medoidsT:
        alreadyMed[cpt].append(element)
        cpt+=1
    
    tetaT=medoidsT
    medoidsS=medoidsT
    CT,costT=makeCluster(medoidsT,matrice1)
    p=0
    medoidsChanged=True
  
    oldMDS=[]
    oldMDT=[]

    while ((medoidsChanged==True) and (p < maxIteration)) :
        p+=1
        tetaS=tetaT
        u=ran(CT,medoidsS)
        labelsSU,sumSU=makeCluster(u,matrice2)
        C,costSMT=makeCluster(medoidsT,matrice2)
        if sumSU < costSMT:
            medoidsS=u
            tetaS=medoidsS
        CS,costS=makeCluster(medoidsS,matrice2)

        p+=1
        tetaT=tetaS
        #print(CS)
        u=ran(CS,medoidsT)
        labelsTU,sumTU=makeCluster(u,matrice1)
        if sumTU < costT:
            medoidsT=u
            tetaT=medoidsT
        CT,costT=makeCluster(medoidsS,matrice1)

        if np.array_equal(oldMDS,medoidsS) and np.array_equal(oldMDT,medoidsT):
            medoidsChanged=False
        oldMDS=medoidsS
        oldMDT=medoidsT

    

    i=0
    for i in range(len(CT)):
        if len(CT[i]) < threshold:
            min_dist=float('inf')
            best_id=-1
            for CTj in CT:
                sum_dist=0
                cnt=0
                j=0
                for user in CT[i]:
                    sum_dist+=matrice1[user][medoidsT[j]]
                    cnt+=1
                j+=1
                if cnt == 0: continue
                avg_dist=sum_dist/cnt
                if min_dist > avg_dist:
                    min_dist=avg_dist
                    best_id=j

            if best_id > -1:
                CT[best_id]=Union(CT[best_id],CT[i])
            CT[i]=[]
            i+=1
    i=0
    for i in range(len(CS)):
        if len(CS[i]) < threshold:
            min_dist=float('inf')
            best_id=-1
            for CSj in CS:
                sum_dist=0
                cnt=0
                j=0
                for user in CS[i]:
                    sum_dist+=matrice2[user][medoidsS[j]]
                    cnt+=1
                j+=1
                if cnt == 0: continue
                avg_dist=sum_dist/cnt
                if min_dist > avg_dist:
                    min_dist=avg_dist
                    best_id=j

            if best_id > -1:
                CS[best_id]=Union(CS[best_id],CS[i])
            CS[i]=[]
            i+=1
    C=CT
    i=0
    for CSj in CS:
        C[i]=Union(C[i],CSj)
        i+=1

    print('medoids similarity=',medoidsS)
    print('medoids semantic=',medoidsT)
    return(C)
#################################################
#START
#K=3
#inItMED=np.random.choice(matrice1.shape[0], K, replace=False)
#inItMED=inItMED.tolist()#must convert to list, not working with np arrays in the functions above

'''
# or you pick
inItMED=[40,0,50,60,70,77,88,900,90,55,64,52,13,11,12,14,15,19,20,100,101,102,103,104,105,106,107,108,109,110,111,112,113] 
#matrice1=trust => #semantique
result = np.loadtxt(open("sem.csv", "rb"), delimiter=",", skiprows=0)
matrice1=result
#matrice2=similarity => #CF
ratings = pd.read_csv("ua.base",sep='\t',names=['user','movie','rating','time'])
usagematrix = ratings.pivot_table(index='user', columns='movie', values='rating')
usagematrix =usagematrix .apply(lambda usagematrix : usagematrix .fillna(usagematrix .mean()), axis=1)
usagearrays=usagematrix.values
matrice2=creatMatrice(usagearrays)


#print("clustering multiview: ",multiView(inItMED,matrice1,matrice2,100,2))
C=multiView(inItMED,matrice1,matrice2,100,5)
#we have two cases: 
#if the user is not in any cluster we do SVD or average prediction 
#else average of clusters


usagematrix = ratings.pivot_table(index='user', columns='movie', values='rating')
usagematrix =usagematrix .apply(lambda usagematrix : usagematrix .fillna(usagematrix .mean()), axis=1)
usagearrays=usagematrix.values

movieDict=createDictTestMovies("ua.test")

#clusters
listelab=C
'''
def CalculePredection(usagearrays,matrice1,matrice2,movieDict,listelab):
    realValue=[]
    predection=[]
    listemeanRAting=meanRatings(usagearrays)
    cpt=0

    for cpt in range(len(usagearrays)):

        liste=functionTest(listelab,cpt)
        listOfMovies=movieDict[cpt]
        if len(liste)==0:
        #faire moyenne des ratings s'il n'est présent dans aucun cluster
            rating=listemeanRAting[cpt]
            for element in listOfMovies:
                for val in element:
                    realValue.append(element[val])
                    predection.append(round(rating))
        
        else:   #if present in one or many cluster i take each rating of each cluster and calculate the average rating
                #the function of prediction is weight sum
            for element in listOfMovies:
                for val in element:
                    realValue.append(element[val])
                    globratin=0
                    for identiCluster in liste:
                        numerateur=0
                        rating=0
                        for user in listelab[identiCluster]:
                            #using both distances in the predection
                            numerateur+=matrice2[cpt][user]*matrice1[cpt][user]*(usagearrays[user][val]-listemeanRAting[user])

                        numerateur=numerateur/(sommeSim(cpt,listelab[identiCluster],matrice1)+sommeSim(cpt,listelab[identiCluster],matrice2))                
                        rating=numerateur+listemeanRAting[cpt]
                        globratin+=rating
                    globratin=globratin/(len(liste))
                    predection.append(round(globratin))

        cpt+=1

    realValue=list(np.float_(realValue))
    mae=mean_absolute_error(realValue,predection)
    rmse=math.sqrt(mean_squared_error(realValue,predection))
    prec = precision_score(realValue, predection, average='micro')
    rec = recall_score(realValue, predection, average='micro')
    
    print("mean_absolute_error and mean_squared_error=",(mae, rmse, prec, rec))
    return (mae, rmse, prec, rec)


#CalculePredection(usagearrays,movieDict,listelab)