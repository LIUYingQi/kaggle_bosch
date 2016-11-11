

import cPickle as pickle
from sklearn import preprocessing
import csv
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

file_path = '/media/liuyingqi/OS/Users/liuyingqi/Desktop/kaggle_project'
D_path = '/media/liuyingqi/Data'

# pre-define value
learning_steps = 1000
input_vec_size = lstm_size = 1
time_step_size = 3108
total_batch = 235
layer1_size = 50
layer2_size = 10
label_size = 2
batch_size = 5000



# load general info from Data file
with open(file_path+'/train_set/trainset1.pkl','rb') as general_info:
    print '1'
    train_batch = pickle.load(general_info)
    # define standard scaler
    scaler = MaxAbsScaler()
    batch_norme = scaler.fit_transform(train_batch)
    general_info.close()

with open(file_path+'/BOSCH/load_code.pkl','rb') as load_code_file:
    load_code = pickle.load(load_code_file)
    print load_code
    print len(load_code)
    load_code_file.close()

with open(file_path + '/BOSCH/test_categorical.csv', 'rb') as cat_info, open(file_path + '/BOSCH/test_numeric.csv', 'rb')as num_info, \
        open(D_path+'/kaggle_test_normeset/testset.csv','wb') as result_file:
    cat_reader = csv.reader(cat_info)
    num_reader = csv.reader(num_info)
    result_writer = csv.writer(result_file)
    print cat_reader.next()
    print num_reader.next()
    for item in range(1183748):
        row_temp = []
        row_cat = cat_reader.next()
        row_cat.append(1)
        row_num = num_reader.next()
        row_num.append(1)
        row_cat.reverse()
        row_num.reverse()
        id = row_cat.pop()
        print id
        row_num.pop()
        for item in range(len(load_code)):
            if load_code[item]==0:
                to_add =row_cat.pop()
                if to_add=='':
                    row_temp.append(0.0)
                elif to_add[0]=='T':
                    row_temp.append(float(to_add[1:]))
            elif load_code[item]==1:
                to_add=row_num.pop()
                if to_add=='':
                    row_temp.append(0.0)
                else:
                    row_temp.append(to_add)
        row_temp = np.array(row_temp,np.float32).reshape((1,3108))
        row_temp = scaler.transform(row_temp)
        np.savetxt(result_file,row_temp,delimiter=',')


