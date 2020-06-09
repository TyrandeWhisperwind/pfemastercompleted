import pandas as pd
import math
import glob
import csv

liste=glob.glob("C:/Users/HP/Desktop/resuults/csv/new/classification/Nouveau dossier/*.csv")
k=[]
rmse=[]
mae=[]

for cpt in liste:
    with open(cpt) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        mot=cpt.split('\\')
        word=mot[1].split(".")
        f= open(word[0]+".txt","w")
        for row in readCSV:
            k.append(row[0])
            rmse.append(row[2])
            mae.append(row[1])
        k.pop(0)
        rmse.pop(0)
        mae.pop(0)
        f.write('\\begin{center} \n \\begin{tabularx}{textwidth}{|l|X|X|X|X|X|X|X|X|X|X|X|X|} \n \\hline \n')
        while k:
            cnt=0
            f.write("K")
            while cnt < 12 :
                if len(k)>=12:
                    f.write(" & "+ k[cnt])
                    print( str(k[cnt]))
                    cnt=cnt+1
                    if cnt==12:
                        f.write(' \\\\ \\hline \n')
                        break
                else: 
                    for element in k:
                        f.write(" & "+ element)
                    f.write('  & & & & & & \\\\ \\hline \n')

                    break
            cnt=0
            f.write("mae")
            while cnt < 12 :
                if len(k)>=12:
                    f.write(" & "+ mae[cnt])
                    cnt=cnt+1
                    if cnt==12:
                        f.write(' \\\\ \\hline \n')
                        break
                else: 
                    for element in mae:
                        f.write(" & "+ element)
                    f.write(' & & & & & &  \\\\ \\hline \n')
                    break
            cnt=0
            f.write("rmse")
            while cnt < 12 :
                if len(k)>=12:
                    f.write(" & "+ rmse[cnt])
                    cnt=cnt+1
                    if cnt==12:
                        f.write(' \\\\ \\hline \n')
                        break
                else: 
                    for element in rmse:
                        f.write(" & "+ element)
                    f.write('  & & & & & & \\\\ \\hline \n')
                    break
            try:
                del mae [0:12]
                del k[0:12]
                del rmse[0:12]
            except: 
                break
        
            if len(k)>1:
                f.write(' \\hline \n')
            else:
                break
                
        f.write('	\\end{tabularx} \n \\captionof{table}{'+word[0].replace('_', ' ' )+ '} \n \\end{center}')
