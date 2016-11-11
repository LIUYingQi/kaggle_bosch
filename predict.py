import cPickle as pickle
from sklearn import preprocessing
import csv
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
import tensorflow as tf
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

# initial for weight
def init_weight(shape):
    return tf.Variable(tf.random_normal(shape,stddev=1.0))

# define model
def model(X,W1,B1,W2,B2,W,B,lstm_size):
    # X,input shape: (batch_size,time_step_size,input_vec_size)
    XT = tf.transpose(X,[1,0,2])
    # XT shape : (time_step_size,batch_size,input_vec_size)
    XR = tf.reshape(XT,[-1,lstm_size])
    # XR shape : (time_step_size * batch_size ,input_vec_size)
    X_split = tf.split(0,time_step_size,XR)
    # sequence_num array with each array(batch_size,input_vec_size )

    # defin lstm cell
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size,forget_bias=1.0,state_is_tuple=True)
    # get lstm cell output
    output , _states = tf.nn.rnn(lstm,X_split,dtype=tf.float32)
    # output : time_step_size arrays with each array (batch_size , LSTM_size) so (time_step_size , batch_size , LSTM_size )

    output = tf.transpose(output,perm=[1,0,2])
    output = tf.reshape(output,[-1,time_step_size*lstm_size])
    # output : (batch_size , time_step_size * LSTM_size)

    # linear activation
    # get the last output

    layer1 = tf.matmul(output,W1) + B1

    layer2 = tf.matmul(layer1,W2) + B2

    # return ( batch_size , 1 )
    return tf.matmul(layer2,W) + B , lstm.state_size

# define X Y
X = tf.placeholder(tf.float32,[None,time_step_size,input_vec_size])
Y = tf.placeholder(tf.float32,[None,label_size])

# get lstm size and output HI

W1 = init_weight([lstm_size*time_step_size,layer1_size])
B1 = init_weight([layer1_size])

W2 = init_weight([layer1_size,layer2_size])
B2 = init_weight([layer2_size])

W = init_weight([layer2_size,label_size])
B = init_weight([label_size])

py_x , state_size = model(X,W1,B1,W2,B2,W,B,lstm_size)

cost = tf.reduce_mean(tf.square( py_x - Y ))
train_op = tf.train.AdamOptimizer(learning_rate=0.009).minimize(cost)
predict_op = tf.argmax(py_x,1)

# saver
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,'model.tfmodel')

    with open('/media/liuyingqi/Data/kaggle_test_normeset/testset.csv', 'rb') as calculate_info, open('result.csv','wb') as result, open('/home/liuyingqi/Desktop/kaggle_project/BOSCH/sample_submission.csv')as sample:
        info_reader = csv.reader(calculate_info)
        sample_reader = csv.reader(sample)
        sample_reader.next()
        result_writer = csv.writer(result, delimiter=',')
        result_writer.writerow(['Id', 'Response'])
        for i in range(1183748):
            print i
            id = np.loadtxt(sample_reader.next(), dtype=np.int, delimiter=',')[0]
            info_predict =  np.loadtxt(info_reader.next(), dtype=np.float32, delimiter=',').reshape((1,3108,1))
            result_writer.writerow([id,sess.run(predict_op,feed_dict={X:info_predict})[0]])