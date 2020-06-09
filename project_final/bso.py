import numpy as np
from typing import Callable
from clustering.kmedoids import kmedoids


class BSO:

    def __init__(self, solution_size, meds, evalution_method: Callable[[list], float], from_count=True):
        self.__solution_size = solution_size
        self.__meds = meds
        if from_count:
            self.__init_sref = self.init_sref_from_count
        else:
            self.__init_sref = self.init_sref_from_meds
        self.__eval = evalution_method
        self.__taboo_list = []
    

    #Generation aleatoire de Sref (vecteur à valeurs entre 0 et 1)
    def init_sref_from_count(self):
        self.sref = np.zeros(self.__solution_size)
        meds = np.random.choice(np.arange(1, self.__solution_size+1, 1), size=(self.__meds))
        
        self.sref[meds] = 1
    
    def init_sref_from_meds(self):
        self.sref = np.zeros(self.__solution_size)
        self.sref[self.__meds] = 1
    
    #Generer les solutions a partir de Sref, par paramètre flip
    def search_areas(self, flip):
        areas = []
        for _ in range(flip):
            areas.append(np.roll(self.sref, 1))
        '''
        h = 0
        while(h < self.__solution_size and h < flip):
            s = np.array(self.sref)
            k = 0
            while(flip * k + h < self.__solution_size):
                s[flip * k + h] = 1 - s[flip * k + h]
                k += 1
            areas.append(s)
            h += 1
        '''
        return areas

    def is_in_taboo(self, arr: np.array):
        for s in self.__taboo_list:
            if np.array_equal(arr, s):
                return True
        
        return False

    #Recherche local dans le voisinage de la solution courrante
    def bee_local_search(self, slocal: np.array, local_max_iter=10):
        min_eval = (10, 10, 10, 10)
        best_solution = slocal
        pos_list = np.nonzero(slocal == 0)[0]
        meds_list = np.nonzero(slocal)[0]
        print(meds_list)
        #print(pos_list)
        pos = -1
        current_solution = np.array(slocal)
        mi = np.random.choice(meds_list)
        current_solution[mi] = 0
        #print(np.count_nonzero(current_solution))
        while(local_max_iter>0):
            i = np.random.choice(pos_list)
            pos_list = np.delete(pos_list, np.where(pos_list == i))
            current_solution[i] = 1 - current_solution[i]
            if self.is_in_taboo(current_solution):
                current_solution[i] = 1 - current_solution[i] # 1 devient 0 et vice versa
                continue
            current_eval = self.__eval(current_solution)
            current_solution[i] = 1 - current_solution[i] # on remet la solution dans l'etat original

            if current_eval[0] < min_eval[0]:
                min_eval = current_eval
                pos = i # sauvgarder la position du meilleur voisinage

            local_max_iter -= 1
        if pos != -1:
            current_solution[pos] = 1 - current_solution[pos]
            best_solution = np.array(current_solution)
        

        return (min_eval,best_solution)
    

    def run(self, flip, global_max_iter = 10, local_max_iter = 10):
        self.__init_sref()
        min_eval = self.__eval(self.sref)

        for _ in range(global_max_iter):
            self.__taboo_list.append(self.sref)

            for area in self.search_areas(flip):
                #print(area)
                current_eval, solution = self.bee_local_search(area, local_max_iter=local_max_iter)
                if current_eval[0] < min_eval[0]:
                    min_eval = current_eval
                    self.sref = solution
        
        return (min_eval, self.sref)

class BSOContinious:

    def __init__(self, epsilon: float, evalution_method: Callable[[float], float]):
        self.epsilon = epsilon
        self.__eval = evalution_method
        self.__taboo_list = []

    #Generation aleatoire de Sref (vecteur à valeurs entre 0 et 1)
    def init_sref(self):
        self.sref = np.random.random_sample()
    
    #Generer les solutions a partir de Sref, par paramètre flip
    def search_areas(self, flip):
        areas = []
        h = 1/flip
        s = self.sref
        while(s >= 0):
            s = s - h
            if s < 0:
                areas.append(0)
                break
            areas.append(s)
        s = self.sref
        while(s < 1):
            s = s + h
            if s >= 1: break
            areas.append(s)
        return areas
        

    #Recherche local dans le voisinage de la solution courrante
    def bee_local_search(self, slocal, max_iter):
        min_eval = self.__eval(slocal)
        best_solution = slocal
        current_solution = slocal
        h = 1/self.flip
        step = h/self.epsilon
         
        while(current_solution < slocal + h and max_iter > 0):
            max_iter -= 1
            current_solution += step
            if current_solution in self.__taboo_list:
                continue
            current_eval = self.__eval(current_solution)
            
            if current_eval < min_eval:
                min_eval = current_eval
                best_solution = current_solution
            
        return (min_eval,best_solution)
    

    def run(self, flip, global_max_iter = 10, local_max_iter = 10):
        self.flip = flip
        self.init_sref()
        min_eval = self.__eval(self.sref)

        for _ in range(global_max_iter):
            self.__taboo_list.append(self.sref)
            areas = self.search_areas(flip)
            print(areas)
            for area in areas:
                (current_eval, solution) = self.bee_local_search(area, local_max_iter)
                if current_eval < min_eval:
                    print(current_eval, solution)
                    min_eval = current_eval
                    self.sref = solution
        
        return (min_eval, self.sref)

'''
def evalution_method_test(num: float):
    return abs(num - 0.4387223454)

alg = BSOContinious(15, evalution_method_test)
mine, sol = alg.run(10, 100, 100)
print(mine, sol)
'''