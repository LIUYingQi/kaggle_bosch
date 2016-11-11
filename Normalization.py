import cPickle as pickle
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
file_path = '/media/liuyingqi/OS/Users/liuyingqi/Desktop/kaggle_project'
count = 0

# load general info from Data file
with open(file_path+'/train_set/trainset1.pkl','rb') as general_info:
    print '1'
    train_batch = pickle.load(general_info)
    # define standard scaler
    scaler = MaxAbsScaler()
    batch_norme = scaler.fit_transform(train_batch)
    batch_norme = np.array(batch_norme,dtype=np.float32)
    with open(file_path + '/train_set/trainset1_norme.pkl', 'wb') as to_save:
        pickle.dump(batch_norme,to_save)

for i in range(235):
    print str(i+2)
    with open(file_path + '/train_set/trainset'+str(i+2)+'.pkl', 'rb') as general_info:
        train_batch = pickle.load(general_info)
        # define standard scaler
        batch_norme = scaler.transform(train_batch)
        batch_norme = np.array(batch_norme, dtype=np.float32)
        with open(file_path + '/train_set/trainset'+str(i+2)+'_norme.pkl', 'wb') as to_save:
            pickle.dump(batch_norme, to_save)

