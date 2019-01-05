# -*- coding: utf-8 -*-


import pandas as pd



dataset = pd.read_csv('solution.csv')

import os
import shutil

path = os.getcwd()

path = path + '\\' + 'training'

os.chdir(path)

os.getcwd()

files = os.listdir(path)

ls = []

final_path_photo = "G:\\training\\"
source = path

for f in files:
    ls.append(f[:-4])

for y in files:    
    for x in range(0,len(dataset)):
        if(y[:-4] == str(dataset.iloc[x][0])):
            new_path  = final_path_photo + str(dataset.iloc[x][1])
            shutil.move(source + "\\" + y,new_path)
            print("working")
            
    
        
    


