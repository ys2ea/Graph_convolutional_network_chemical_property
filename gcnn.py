#Use a simple GCNN combined with multi-task learning

import numpy as np
import tensorflow as tf
from mol_feature import load_data, circular_fps, build_feature

fp_dim = 128
output_dim = 12
hidden_l = 64
lambda_loss = 0.2
batch_size = 64
step = 50000
length_atom_feature = 53
length_bond_feature = 5
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
    entropy = -balance_weight * mlabel * tf.log(tf.clip_by_value(mpredict,1e-9,1.0)) - (1-mlabel) * tf.log(tf.clip_by_value((1-predict),1e-9,1.0))
    return tf.reduce_sum(entropy)
    
def mse_missing_values(predict, label):
    valid = tf.logical_not(tf.is_nan(label))    
    other = tf.zeros_like(label)
    mpredict = tf.where(valid, predict, other) 
    mlabel = tf.where(valid, label, other)    
    loss = tf.reduce_sum(tf.square(mpredict-mlabel))    
    return loss
    
def accuracy(predict, label, threshold):
    valid = np.logical_not(np.isnan(label))
    predict_fl = np.array(np.greater(predict, threshold), dtype=float)
    
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
    
def graph_convolutino(atom_feature, bond_feature, connection, W, b):
    concat = tf.concat([atom_feature, bond_feature], axis=1)
    return tf.nn.relu(tf.matmul(concat, W) + b)


##placeholder for the inputs needed: atom feature array, bond featuer array, connection map, training labels
    
af = tf.placeholder('float', [None, length_atom_feature])

bf = tf.placeholder('float', [None, length_bond_feature])

cf = tf.placeholder('float', [None, None])

Y = tf.placeholder('float', [None, output_dim])

## defind 3 simple graph convolutional layers, more sophisticated convolution to be added later

cW1 = tf.Variable(tf.random_normal([length_atom_feature + length_bond_feature, length_atom_feature], 0, 0.1))
cb1 = tf.Variable(tf.random_normal([length_atom_feature], 0, 0.1))

cW2 = tf.Variable(tf.random_normal([length_atom_feature + length_bond_feature, length_atom_feature], 0, 0.1))
cb2 = tf.Variable(tf.random_normal([length_atom_feature], 0, 0.1))

cW3 = tf.Variable(tf.random_normal([length_atom_feature + length_bond_feature, length_atom_feature], 0, 0.1))
cb3 = tf.Variable(tf.random_normal([length_atom_feature], 0, 0.1))

## create fingerprints from gc

fW = tf.Variable(tf.random_normal([length_atom_feature, fp_dim], 0, 0.1))
fb = tf.Variable(tf.random_normal([fp_dim], 0, 0.1))

## add another 2 fully connected layers 

hw1 = tf.Variable(tf.random_normal([fp_dim, hidden_l], 0, 0.1))
hb1 = tf.Variable(tf.random_normal([hidden_l], 0, 0.1))
    
hw2 = tf.Variable(tf.random_normal([hidden_l, output_dim], 0, 0.1))
hb2 = tf.Variable(tf.random_normal([output_dim], 0, 0.1))
    
c_layer1 = graph_convolutino(af, bf, cf, cW1, cb1)

c_layer2 = graph_convolutino(c_layer1, bf, cf, cW2, cb2)

c_layer3 = graph_convolutino(c_layer2, bf, cf, cW3, cb3)
    
fp = tf.reduce_sum(tf.nn.softmax(tf.matmul(c_layer3, fW) + fb), axis=0, keepdims=True)

h_layer1 = tf.nn.relu(tf.matmul(fp, hw1) + hb1)
    
predict = tf.nn.sigmoid(tf.matmul(h_layer1, hw2) + hb2)

predict = tf.reshape(predict, [1, -1])

l2 = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

entropy = cross_entropy_missing_values(predict, Y)

loss = entropy + l2

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

X_train = smiles[0:6000]
Y_train = labels[0:6000]

X_test = smiles[6000:]
Y_test = labels[6000:,:]

data_statistics(labels)
data_statistics(Y_test)


with tf.Session() as sess:
     tf.global_variables_initializer().run()
     for i in range(step):
         offset = i % X_train.shape[0]
         batch_x = X_train[offset]
         batch_y = Y_train[offset:offset+1, :]
         
         [batch_af, batch_bf, batch_cf] = build_feature(batch_x)
         feed_dict = {af : batch_af, bf : batch_bf, cf : batch_cf, Y : batch_y}
         _, p, error = sess.run([train_step, predict, entropy], feed_dict=feed_dict)
         
         if i%1000 == 0:
             
             #_, test_p = sess.run([entropy, predict], feed_dict={X:X_test, Y:Y_test})
             #t_error = accuracy(test_p, Y_test, 0.5)
             #print(p[0])
             #print(batch_y[0])
             np.set_printoptions(precision=3)
             
             print("step: {}, train error: {}, test error: ".format(i, error))
             #print("TPR: ", t_error[0])
             #print("FPR: ", t_error[1])
    
