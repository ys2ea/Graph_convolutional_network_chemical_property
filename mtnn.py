#Use a simple multi-task nn, without graph convolution.

import numpy as np
import tensorflow as tf
from mol_feature import load_data, circular_fps

input_dim = 512
output_dim = 12
hidden_l = 64
lambda_loss = 0.05
batch_size = 40
step = 20000
balance_weight = np.array([10.5, 15.3,  4.4,  9.6,  3.8,  8.4, 18.2,  3.2, 14.8, 8.7,  3.3,  9.5])

[smiles, labels] = load_data('data.csv') 
#labels = np.nan_to_num(labels)



def data_statistics(Y_test):
    tvalid = np.logical_not(np.isnan(Y_test))

    total_pos = np.where(tvalid, Y_test, 0)
    print("Total: ", np.sum(tvalid, axis=0))
    total_neg = np.where(tvalid, (1-Y_test), 0)
    rate = np.sum(total_pos,axis=0) / ( np.sum(total_pos,axis=0) +  np.sum(total_neg,axis=0))
    print("Pos rate: ------", rate)
    print("weights: ", 0.5/rate)
    
def cross_entropy_missing_values(predict, label):
    valid = tf.logical_not(tf.is_nan(label))
    other = tf.zeros_like(predict)   
    mpredict = tf.where(valid, predict, other)
    mlabel = tf.where(valid, label, other) 
    #entropy = tf.where(tf.is_nan(label), other, -label * tf.log(tf.clip_by_value(predict,1e-9,1.0)) \
        #- (1-label) * tf.log(tf.clip_by_value(1-predict,1e-9,1.0))) 
    entropy = -balance_weight * mlabel * tf.log(tf.clip_by_value(mpredict,1e-9,1.0)) - (1-mlabel) * tf.log(tf.clip_by_value((1-predict),1e-9,1.0))
    return tf.reduce_sum(entropy)
    
def mse_missing_values(predict, label):
    valid = tf.logical_not(tf.is_nan(label))    
    other = tf.zeros_like(label)
    mpredict = tf.where(valid, predict, other) 
    mlabel = tf.where(valid, label, other)    
    loss = tf.reduce_sum(tf.square(mpredict-mlabel))    
    return loss

def accuracy_tf(predict, label, threshold):
    valid = tf.logical_not(tf.is_nan(label))
    predict_fl = tf.cast(tf.greater(predict, threshold), tf.float32)
    
    #total_count = tf.reduce_sum(valid, axis=0)
    total_pos = tf.reduce_sum(tf.cast(valid, tf.float32) * label)
    total_neg = tf.reduce_sum(tf.cast(valid, tf.float32) * (1-label))
    
    true_pos = tf.reduce_sum(tf.cast(valid, tf.float32) * label * predict_fl)
    true_neg = tf.reduce_sum(tf.cast(valid, tf.float32) * (1-label) * (1-predict_fl))
    false_pos = tf.reduce_sum(tf.cast(valid, tf.float32) * (1-label) * predict_fl)
    false_neg = tf.reduce_sum(tf.cast(valid, tf.float32) * label * (1-predict_fl))
    
    return [true_pos / (true_pos + false_neg), false_pos / (true_neg + false_pos)]
    
    
def accuracy(predict, label, threshold):
    valid = np.logical_not(np.isnan(label))
    predict_fl = np.array(np.greater(predict, threshold), dtype=float)
    
    #total_count = np.where(valid, label, 0)
    total_true = np.where(valid, label * predict_fl + (1-label) * (1-predict_fl), 0)
    
    
    true_pos = np.where(valid, label * predict_fl, 0)
    true_neg = np.where(valid, (1-label) * (1-predict_fl), 0)
    false_pos = np.where(valid, (1-label) * predict_fl, 0)
    false_neg = np.where(valid, label * (1-predict_fl), 0)
    
    sum_tp = np.sum(true_pos, axis=0)
    sum_tn = np.sum(true_neg, axis=0)
    sum_fp = np.sum(false_pos, axis=0)
    sum_fn = np.sum(false_neg, axis=0)
    #print(predict_fl)
    #print(label)
    return [sum_tp / (sum_tp + sum_fn), sum_fp / (sum_tn + sum_fp)]
    #return np.sum(total_true) / np.sum(valid)
    
X = tf.placeholder('float', [None, input_dim])

Y = tf.placeholder('float', [None, output_dim])

hw1 = tf.Variable(tf.random_normal([input_dim, hidden_l], 0, 0.1))
hb1 = tf.Variable(tf.random_normal([hidden_l], 0, 0.1))
    
hw2 = tf.Variable(tf.random_normal([hidden_l, hidden_l], 0, 0.1))
hb2 = tf.Variable(tf.random_normal([hidden_l], 0, 0.1))
    
hw3 = tf.Variable(tf.random_normal([hidden_l, output_dim], 0, 0.1))
hb3 = tf.Variable(tf.random_normal([output_dim], 0, 0.1))
    
h_layer1 = tf.nn.relu(tf.matmul(X, hw1) + hb1)
    
h_layer2 = tf.nn.relu(tf.matmul(h_layer1, hw2) + hb2)
    
predict = tf.sigmoid(tf.matmul(h_layer2, hw3) + hb3)
    
l2 = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

entropy = cross_entropy_missing_values(predict, Y)
#entropy = mse_missing_values(predict, Y)
#entropy = -tf.reduce_sum(Y * tf.log(tf.clip_by_value(predict,1e-10,1.0)) + (1-Y) * tf.log(tf.clip_by_value(1-predict,1e-10,1.0)))
loss = entropy + l2

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
X_train = circular_fps(smiles[0:6000])
Y_train = labels[0:6000]

X_test = circular_fps(smiles[6000:])
Y_test = labels[6000:,:]

data_statistics(labels)
data_statistics(Y_test)


with tf.Session() as sess:
     tf.global_variables_initializer().run()
     for i in range(step):
         offset = (i * batch_size) % X_train.shape[0]
         batch_x = X_train[offset:offset+batch_size]
         batch_y = Y_train[offset:offset+batch_size]
         
         feed_dict = {X:batch_x, Y:batch_y}
         
         _, p, error = sess.run([train_step, predict, entropy], feed_dict=feed_dict)
         
         if i%200 == 0:
             
             _, test_p = sess.run([entropy, predict], feed_dict={X:X_test, Y:Y_test})
             t_error = accuracy(test_p, Y_test, 0.5)
             #print(p[0])
             #print(batch_y[0])
             np.set_printoptions(precision=3)
             
             print("step: {}, train error: {}, test error: ".format(i, error))
             print("TPR: ", t_error[0])
             print("FPR: ", t_error[1])
    
