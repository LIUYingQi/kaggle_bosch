import csv
import cPickle as pickle
import numpy as np
import os

file_path = '/media/liuyingqi/OS/Users/liuyingqi/Desktop/kaggle_project'
batch_size = 5000

count=0
for i in range(300):
    file_name = '' + str(i + 1) + '.pkl'
    try:
        os.remove(file_path+'/train_set/trainset'+str(i+1)+'_label.pkl')
    except OSError:
        print 'no such files  Test set  ----   continue  '
    else:
        print 'delete old file  Test  set  ----   continue  '

with open(file_path+'/BOSCH/train_numeric.csv','rb') as cat_file:
    reader = csv.reader(cat_file)
    print reader.next()[-1]
    for i in range(236):
        count = 0
        with open(file_path+'/train_set/trainset'+str(i+1)+'_label.pkl','wb') as to_label:
            temp = np.empty((0,2),dtype=np.float32)
            case_0 = np.array([200.0,0.0],dtype=np.float32).reshape(1,2)
            case_1 = np.array([0.0,2000.0],dtype=np.float32).reshape(1,2)
            for j in range(batch_size):
                if (reader.next()[-1])== '0':
                    temp = np.concatenate((temp,case_0),axis=0)
                else:
                    temp = np.concatenate((temp, case_1), axis=0)
                    count+=1

            pickle.dump(temp,to_label)
            print count


