import csv
import numpy as np

file_path = '/media/liuyingqi/OS/Users/liuyingqi/Desktop/kaggle_project'
count = 0

# load general info from Data_strcture.csv file
with open('result.csv','rb') as general_info:
    data = csv.reader(general_info)
    for row in data:
        if row[1]=='1':
            count+=1
            print row
print count