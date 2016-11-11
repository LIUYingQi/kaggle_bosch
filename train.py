# this file is to do a regression to predict RUL

####################################################################################
# import and useful function
####################################################################################

import tensorflow as tf
import numpy as np
import cPickle as pickle
import time

# start time
start_time = time.time()

# data file path
data_file_path = '/media/liuyingqi/OS/Users/liuyingqi/Desktop/kaggle_project/'

########################################################################################
# define model
########################################################################################

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


##################################################################################
# def process
##################################################################################

def train_part(total_batch,batch_size,time_step_size,input_vec_size,label_size):
    print '-------------- training -----------------'

    # train for all batch for each steps but with some seperate batch
    for item_train in range(total_batch):

        with open(data_file_path+'train_set/trainset'+str(item_train+1)+'_norme.pkl' , 'rb') as dataset_info_to_load:
            one_batch = pickle.load(dataset_info_to_load)
            one_batch = np.array(one_batch,dtype=np.float32)
            dataset_info_to_load.close()

        with open(data_file_path+ 'train_set/trainset' + str(item_train + 1) + '_label.pkl','rb') as dataset_info_to_load:
            one_label = pickle.load(dataset_info_to_load)
            one_label = np.array(one_label,dtype=np.float32)
            dataset_info_to_load.close()

        print 'sess run for step :' + str(step + 1) + ' /  batch : ' + str(item_train + 1)
        print 'input train batch: ' + str(one_batch.shape) + '----' + str(one_label.shape)

        one_batch = np.reshape(one_batch,(batch_size,time_step_size,input_vec_size))
        one_label = np.reshape(one_label,(batch_size,label_size))
        # train process
        _ , loss = sess.run([train_op,cost], feed_dict={X: one_batch, Y: one_label})
        print loss

def write_part(total_batch,batch_size,time_step_size,input_vec_size,label_size):
    print '----------------- testing -----------------'

###################################################################################
# lunch session
###################################################################################

with tf.Session() as sess:

    # model value initialization
    tf.initialize_all_variables().run()

    # learning_steps
    for step in range(learning_steps):

        print '                                                            '
        print '############################################################'
        print '                                                            '

        # show time
        end_time = time.time()
        cost_time = end_time - start_time
        print ' total run time : ' + str(cost_time)

        # show step info
        print 'step : ' + str(step+1) + ' in ' + str(learning_steps) + ' learning steps'

        # train part
        train_part(total_batch,batch_size,time_step_size,input_vec_size,label_size)

        # write result
        write_part(total_batch, batch_size, time_step_size, input_vec_size, label_size)

        saver.save(sess, 'model.tfmodel')

