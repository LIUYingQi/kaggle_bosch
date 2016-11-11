import cPickle as pickle
file_path = '/media/liuyingqi/OS/Users/liuyingqi/Desktop/kaggle_project'
count = 0

# load general info from Data_strcture.csv file
with open(file_path+'/train_set/trainset3_norme.pkl','rb') as general_info:
    data = pickle.load(general_info)
    for row in data:
        print row

