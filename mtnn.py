#Use a simple multi-task nn, without graph convolution.

import numpy as np
import tensorflow as tf
from mol_feature import load_data, circular_fps
import matplotlib.pyplot as plt

input_dim = 1024
output_dim = 12
hidden_l = 64
lambda_loss = 1.2
batch_size = 40
step = 200000
keepprob = 0.6
lr = 0.005
lr1 = 0.0005
save_curve = False
balance_weight = np.array([10.5, 15.3,  4.4,  9.6,  3.8,  8.4, 18.2,  3.2, 14.8, 8.7,  3.3,  9.5])

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

    return [sum_tp / (sum_tp + sum_fn), sum_fp / (sum_tn + sum_fp)]
    
def ROC_curve(predict, label):
    [tp, fp] = accuracy(predict, label, 0)
    
    thresh = 0.0000000002
    for i in range(8):
        [ttp, tfp] = accuracy(predict, label, thresh)
        tp = np.vstack((tp, ttp))
        fp = np.vstack((fp, tfp))
        thresh *= 10
        
    for t in range(1,50):
        thresh = t/50.
        [ttp, tfp] = accuracy(predict, label, thresh)
        tp = np.vstack((tp, ttp))
        fp = np.vstack((fp, tfp))
    
    
    
    res = []
    ave = 0.
    for i in range(12):
        c = auc(fp[:,i],tp[:,i])
        ave += c
        res.append(c)
    
    res.append(ave/12.)
    return np.array(res)
    #plt.plot(fp[:,0],tp[:,0])
    #plt.show()
    

def auc(fp, tp):
    result = 0.
    for i in range(fp.shape[0]-1): 
        result += (fp[i]-fp[i+1]) * (tp[i]+tp[i+1])/2.
    return result
                
X = tf.placeholder('float', [None, input_dim])

Y = tf.placeholder('float', [None, output_dim])

hw1 = tf.Variable(tf.random_normal([input_dim, hidden_l], 0, 0.1))
hb1 = tf.Variable(tf.random_normal([hidden_l], 0, 0.1))
    
#hw2 = tf.Variable(tf.random_normal([hidden_l, hidden_l], 0, 0.1))
#hb2 = tf.Variable(tf.random_normal([hidden_l], 0, 0.1))
    
#hw3 = tf.Variable(tf.random_normal([hidden_l, hidden_l], 0, 0.1))
#hb3 = tf.Variable(tf.random_normal([hidden_l], 0, 0.1))
    
hw4 = tf.Variable(tf.random_normal([hidden_l, output_dim], 0, 0.1))
hb4 = tf.Variable(tf.random_normal([output_dim], 0, 0.1))
    
h_layer1 = tf.nn.relu(tf.matmul(X, hw1) + hb1)
h_layer1 = tf.nn.dropout(h_layer1, keepprob)   
 
#h_layer2 = tf.nn.relu(tf.matmul(h_layer1, hw2) + hb2)
#h_layer2 = tf.nn.dropout(h_layer2, keepprob) 
    
#h_layer3 = tf.nn.relu(tf.matmul(h_layer2, hw3) + hb3)
#h_layer3 = tf.nn.dropout(h_layer3, keepprob) 
    
predict = tf.sigmoid(tf.matmul(h_layer1, hw4) + hb4)
    
l2 = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

entropy = cross_entropy_missing_values(predict, Y)

loss = entropy + l2

train_step1 = tf.train.AdamOptimizer(lr).minimize(loss)
train_step2 = tf.train.GradientDescentOptimizer(lr1).minimize(loss)
 
[smiles, labels] = load_data('data.csv') 
X_train = circular_fps(smiles[0:7000])
Y_train = labels[0:7000]

X_test = circular_fps(smiles[7000:])
Y_test = labels[7000:,:]

log_path = 'logdir'
train_writer = tf.summary.FileWriter(log_path, tf.get_default_graph())



f = open("data_mtnn.dat", "w")

with tf.Session() as sess:
     tf.global_variables_initializer().run()
     for i in range(step):
         offset = (i * batch_size) % X_train.shape[0]
         batch_x = X_train[offset:offset+batch_size]
         batch_y = Y_train[offset:offset+batch_size]
         
         feed_dict = {X:batch_x, Y:batch_y}
         
         if i < 4000:
             _, p, error = sess.run([train_step1, predict, entropy], feed_dict=feed_dict)
         else:
             _, p, error = sess.run([train_step2, predict, entropy], feed_dict=feed_dict)
         

         if save_curve and i%20 == 0 : 

             train_e, train_p = sess.run([entropy, predict], feed_dict={X:X_train, Y:Y_train})
             test_e, test_p = sess.run([entropy, predict], feed_dict={X:X_test, Y:Y_test})
             auc_train = ROC_curve(train_p,Y_train)[-1]
             auc_test = ROC_curve(test_p,Y_test)[-1]
             f.write("{}, {}, {}, {}, {} \n".format(i, train_e/X_train.shape[0], auc_train, test_e/X_test.shape[0], auc_test))
             
         if i%2000 == 0:
             
             test_e, test_p = sess.run([entropy, predict], feed_dict={X:X_test, Y:Y_test})
             train_e, _ = sess.run([entropy, predict], feed_dict={X:X_train, Y:Y_train})
             t_error = ROC_curve(test_p, Y_test)
             np.set_printoptions(precision=3)
             
             print("step: {}, train error: {}, test error: {} ".format(i, train_e/X_train.shape[0], test_e/X_test.shape[0]))
             print(t_error)
         
    
