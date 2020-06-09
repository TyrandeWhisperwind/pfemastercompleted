import pandas as pd
import math
import glob
liste=glob.glob("C:/Users/HP/Desktop/resuults/csv/*.csv")
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    
    return math.trunc(stepper * number) / stepper
###################################################################
def try_cutoff(x):

    try:
        return truncate(x, 3) 
    except Exception:
        return x
##############################################################################""
for cpt in liste:
    dataset = pd.read_csv(cpt)

    for field in dataset.columns:


        dataset[field] = dataset[field].map(try_cutoff)

    mot=cpt.split('\\')
    dataset.to_csv(mot[1]+"_new.csv", index = False)