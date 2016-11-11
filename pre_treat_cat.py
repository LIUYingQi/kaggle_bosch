import csv
import cPickle as pickle

file_path = '/home/liuyingqi/Desktop/kaggle_project'
count = 0

cat_type=[]

# load general info from Data_strcture.csv file

with open(file_path+'/BOSCH/train_categorical.csv','rb') as cat_info:
    reader = csv.reader(cat_info)
    reader.next()
    for row in reader:
        row = row[1:]
        for item in row:
            if item!='' and item not in cat_type:
                cat_type.append(item)

print cat_type