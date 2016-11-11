import csv
import cPickle as pickle
import numpy as np
import os

file_path = '/home/liuyingqi/Desktop/kaggle_project'
count = 0

with open(file_path+'/BOSCH/load_code.pkl','rb') as load_code_file:
    load_code = pickle.load(load_code_file)
    print load_code
    print len(load_code)

for i in range(150):
    file_name = '' + str(i + 1) + '.pkl'
    try:
        os.remove(file_path+'/train_set/trainset'+str(i+1)+'.pkl')
    except OSError:
        print 'no such files  Test set  ----   continue  '
    else:
        print 'delete old file  Test  set  ----   continue  '

with open(file_path + '/BOSCH/train_categorical.csv', 'rb') as cat_info, open(file_path + '/BOSCH/train_numeric.csv', 'rb')as num_info:
    cat_reader = csv.reader(cat_info)
    num_reader = csv.reader(num_info)
    print cat_reader.next()
    print num_reader.next()
    batch_size = 10000
    for batch_item in range(1183748/batch_size):
        with open(file_path+'/train_set/trainset'+str(batch_item+1)+'.pkl','wb') as set_to_save:
            temp = np.empty((0,len(load_code)),np.float32)

            for row_item in range(batch_size):
                row_temp = []
                row_cat = cat_reader.next()
                print row_cat
                row_cat.append(1)
                row_num = num_reader.next()
                print row_num
                row_cat.reverse()
                row_num.reverse()
                row_cat.pop()
                row_num.pop()
                for item in range(len(load_code)):
                    if load_code[item]==0:
                        to_add =row_cat.pop()
                        if to_add=='':
                            row_temp.append(0.0)
                            print '0.0'
                        elif to_add[0]=='T':
                            row_temp.append(float(to_add[1:]))
                            print float(to_add[1:])
                    elif load_code[item]==1:
                        to_add=row_num.pop()
                        if to_add=='':
                            row_temp.append(0.0)
                            print '0.0'
                        else:
                            row_temp.append(to_add)
                            print to_add
                row_temp = np.array(row_temp,np.float32).reshape((1,len(load_code)))
                print '###########################################################'
                print row_temp.shape
                print temp.shape
                print np.append(temp,row_temp,axis=0).shape
            pickle.dump(temp,set_to_save)