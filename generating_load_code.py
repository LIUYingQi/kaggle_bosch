import csv
import cPickle as pickle
import numpy as np

file_path = '/home/liuyingqi/Desktop/kaggle_project'
count = 0

# load general info from Data_strcture.csv file
with open(file_path+'/BOSCH/train_categorical.csv','rb') as cat_info,open(file_path+'/BOSCH/train_numeric.csv','rb')as num_info:
    cat_reader = csv.reader(cat_info)
    num_reader = csv.reader(num_info)
    cat_list = cat_reader.next()[1:]
    num_list = num_reader.next()[1:]
    print len(cat_list)
    print len(num_list)
    uni_list = []
    print 'categore list : '
    print cat_list
    print 'numeric list : '
    print num_list
    cat_list.append('Responsa')
    cat_list.reverse()
    num_list.reverse()
    cat_now = cat_list.pop()
    num_now = num_list.pop()
    while not(len(cat_list)==0 and len(num_list)==0):
        if cmp(cat_now,num_now)<0 :
            if len(cat_list)!=0:
                print cat_now
                cat_now = cat_list.pop()
                uni_list.append(0)
            else:
                print num_now
                num_now = num_list.pop()
                uni_list.append(1)
        elif cmp(cat_now,num_now)>0 :
            if len(num_list)!=0:
                print num_now
                num_now = num_list.pop()
                uni_list.append(1)
            else :
                print cat_now
                cat_now = cat_list.pop()
                uni_list.append(0)
    print uni_list
    print len(uni_list)

load_code = np.array(uni_list,np.int8)
with open(file_path+'/BOSCH/load_code.pkl','wb') as load_code_file:
    pickle.dump(load_code,load_code_file)