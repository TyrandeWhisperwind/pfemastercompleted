import csv
import numpy as np
import pandas as pd
import sys



class DataSet:

    def __init__(self, user_ratings_path: str, users_path:str , movies_path: str):
        self.__usage_matrix, self.__movie_matrix, self.__ratings = self.load_data(user_ratings_path, users_path, movies_path)
    
    def get_usage_matrix(self):
        return self.__usage_matrix
        
    def set_usage_matrix(self, matrix):
        self.__usage_matrix = matrix
    
    def get_movie_matrix(self):
        return self.__movie_matrix

    def get_ratings(self):
        return self.__ratings
    

    def load_data(self, user_ratings_path: str, users_path:str , movies_path: str):
            ratings = np.array(
                list(csv.reader(open(user_ratings_path, "r", encoding="ISO-8859-1"),delimiter='\t'))
            ).astype('int')

            #ratings = pd.read_csv(user_ratings_path,sep='\t',names=['user','movie','rating','time'])
            #usagematrix = ratings.pivot_table(index='user', columns='movie', values='rating')
            #print(type(usagematrix))
            movies = np.array(
                [
                    row[5:]
                    for row in csv.reader(open(movies_path, "r", encoding="ISO-8859-1"),delimiter='|')
                ]
            ).astype('int')

            
            users = np.array(
                list(
                    csv.reader(open(users_path, "r", encoding="ISO-8859-1"),delimiter='\t')
                )
            )

            usage_matrix = np.zeros((len(users), len(movies)))
            for rating in ratings:
                usage_matrix[rating[0]-1, rating[1]-1] = rating[2]
            
            #print(len(usage_matrix[0]), usage_matrix[0])
            #print(len(movies), movies[0])
            
            return (usage_matrix, movies, ratings)


def dump_array(arr: np.array, filename, sep=';', fmt='%.18e'):
    np.savetxt(filename, arr, delimiter=sep, fmt=fmt)

def load_data(filename, sep=';') ->np.array:
    return np.loadtxt(filename, delimiter=sep)

#load_data('/home/imad/Desktop/PFE/db/ml-100k/u.data', '/home/imad/Desktop/PFE/db/ml-100k/u.user', '/home/imad/Desktop/PFE/db/ml-100k/u.item')
