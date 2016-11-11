import cPickle as pickle
from sklearn import preprocessing
import csv
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

file_path = '/media/liuyingqi/OS/Users/liuyingqi/Desktop/kaggle_project'


# pre-define value
learning_steps = 1000
input_vec_size = lstm_size = 1
time_step_size = 3108
total_batch = 235
layer1_size = 50
layer2_size = 10
label_size = 2
batch_size = 5000


# test result list
accuracy_list = []
accuracy_nominal_case_list= []
accuracy_fault_case_list = []



with open('/media/liuyingqi/Data/kaggle_test_normeset/testset.csv', 'rb') as calculate_info,open('result.csv', 'wb') as result,open('/home/liuyingqi/Desktop/kaggle_project/BOSCH/sample_submission.csv')as sample:
    info_reader = csv.reader(calculate_info)
    sample_reader = csv.reader(sample)
    sample_reader.next()
    result_writer = csv.writer(result,delimiter=',')
    result_writer.writerow(['Id','Response'])
    for i in range (1183748):
        print np.loadtxt(sample_reader.next(),dtype=np.int,delimiter=',')[0]
        print np.loadtxt(info_reader.next(),dtype=np.float32,delimiter=',')
        # result_writer.writerow([id,sess.run(predict_op,feed_dict={X:row_temp})[0]])
