#Use a simple GCNN combined with multi-task learning
#Also use a simple padding method for mini-batching

import numpy as np
import random
import tensorflow as tf
from mol_feature import load_data, circular_fps, build_feature, build_batch_feature

fp_dim = 128
output_dim = 12
hidden_l = 64
lambda_loss = 0.1
batch_size = 32
step = 120000
length_atom_feature = 53
length_bond_feature = 5
lr = 0.005
lr1 = 0.002
keepprob = 0.7
save_curve = False
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
    
def graph_convolutino(atom_feature, bond_feature, connection, W, b):
    concat = tf.concat([atom_feature, bond_feature], axis=2)
    return tf.nn.relu(tf.tensordot(concat, W, axes=[[2],[0]]) + b)

#def graph_convolutino(atom_feature, bond_feature, connection, W):
#    concat = tf.concat([atom_feature, bond_feature], axis=2)
#    return tf.nn.relu(tf.tensordot(concat, W, axes=[[2],[0]]))

##placeholder for the inputs needed: atom feature array, bond featuer array, connection map, training labels
    
af = tf.placeholder('float', [None, None, length_atom_feature])

bf = tf.placeholder('float', [None, None, length_bond_feature])

cf = tf.placeholder('float', [None, None, None])

Y = tf.placeholder('float', [None, output_dim])

## defind simple graph convolutional layers, more sophisticated convolution to be added later

cW1 = tf.Variable(tf.random_normal([length_atom_feature + length_bond_feature, length_atom_feature], 0, 0.06))
cb1 = tf.Variable(tf.zeros([length_atom_feature]))

cW2 = tf.Variable(tf.random_normal([length_atom_feature + length_bond_feature, length_atom_feature], 0, 0.12))
cb2 = tf.Variable(tf.zeros([length_atom_feature]))

#cW3 = tf.Variable(tf.random_normal([length_atom_feature + length_bond_feature, length_atom_feature], 0, 0.1))
#cb3 = tf.Variable(tf.zeros([length_atom_feature]))

#cW4 = tf.Variable(tf.random_normal([length_atom_feature + length_bond_feature, length_atom_feature], 0, 0.1))
#cb4 = tf.Variable(tf.zeros([length_atom_feature]))

## create fingerprints from gc

fW = tf.Variable(tf.random_normal([length_atom_feature, fp_dim], 0, 0.1))
fb = tf.Variable(tf.random_normal([fp_dim], 0, 0.1))

## add another 2 fully connected layers 

hw1 = tf.Variable(tf.random_normal([fp_dim, hidden_l], 0, 0.1))
hb1 = tf.Variable(tf.random_normal([hidden_l], 0, 0.1))
    
#hw2 = tf.Variable(tf.random_normal([hidden_l, hidden_l], 0, 0.1))
#hb2 = tf.Variable(tf.random_normal([hidden_l], 0, 0.1))
    
hw3 = tf.Variable(tf.random_normal([hidden_l, output_dim], 0, 0.1))
hb3 = tf.Variable(tf.random_normal([output_dim], 0, 0.1))
    
c_layer1 = graph_convolutino(af, bf, cf, cW1, cb1)
#c_layer1 = tf.nn.dropout(c_layer1, keepprob)

c_layer2 = graph_convolutino(c_layer1, bf, cf, cW2, cb2)
#c_layer2 = tf.nn.dropout(c_layer2, keepprob)

#c_layer3 = graph_convolutino(c_layer2, bf, cf, cW3)
#c_layer3 = tf.nn.dropout(c_layer3, keepprob)

#c_layer4 = graph_convolutino(c_layer3, bf, cf, cW4)
#c_layer4 = tf.nn.dropout(c_layer4, keepprob)

gcout = tf.nn.softmax(tf.tensordot(c_layer2, fW, axes=[[2],[0]]) + fb, axis = 2)

fp = tf.reduce_sum(gcout, axis=1)

h_layer1 = tf.nn.relu(tf.matmul(fp, hw1) + hb1)
#h_layer1 = tf.nn.dropout(h_layer1, keepprob)

predict = tf.nn.sigmoid(tf.matmul(h_layer1, hw3) + hb3)

l2 = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

entropy = cross_entropy_missing_values(predict, Y)

loss = entropy + l2

#calculate atom activation
#only run this for batch size 1
aa1 = tf.nn.relu(tf.tensordot(gcout, hw1, axes=[[2],[0]]) + hb1)
aa2 = tf.tensordot(aa1, hw3, axes=[[2],[0]]) + hb3

atom_activation = tf.nn.sigmoid(aa2)

train_step1 = tf.train.AdamOptimizer(lr).minimize(loss)
train_step2 = tf.train.GradientDescentOptimizer(lr1).minimize(loss)

X_train = smiles[0:7000]
Y_train = labels[0:7000]

X_test = smiles[7000:]
Y_test = labels[7000:,:]

#data_statistics(labels)
#data_statistics(Y_test)
f = open("data_gcnn.dat", "w")
idx = [i for i in range(X_train.shape[0])]

with tf.Session() as sess:
     tf.global_variables_initializer().run()
     for i in range(step):
         offset = (i * batch_size) % X_train.shape[0]
         batch_x = X_train[offset:offset+batch_size]
         batch_y = Y_train[offset:offset+batch_size,:]
         
         [batch_af, batch_bf, batch_cf] = build_batch_feature(batch_x)
         
         feed_dict = {af : batch_af, bf : batch_bf, cf : batch_cf, Y : batch_y}
         if i < 80000:
             _, p, error = sess.run([train_step1, predict, entropy], feed_dict=feed_dict)
         else:
             _, p, error = sess.run([train_step2, predict, entropy], feed_dict=feed_dict)
             
         if save_curve and i%20 == 0 : 
         
             t_idx = random.sample(idx, 2500)
             [train_af, train_bf, train_cf] = build_batch_feature(X_train[t_idx])
             [test_af, test_bf, test_cf] = build_batch_feature(X_test)
             train_e, train_p = sess.run([entropy, predict], feed_dict={af:train_af, bf:train_bf, cf:train_cf, Y:Y_train[t_idx]})
             test_e, test_p = sess.run([entropy, predict], feed_dict={af:test_af, bf:test_bf, cf:test_cf, Y:Y_test})
             auc_train = ROC_curve(train_p,Y_train[t_idx])[-1]
             auc_test = ROC_curve(test_p,Y_test)[-1]
             f.write("{}, {}, {}, {}, {} \n".format(i, train_e/2500., auc_train, test_e/X_test.shape[0], auc_test))
             f.flush()
             
         if i%1000 == 0:

             [test_af, test_bf, test_cf] = build_batch_feature(X_test)            
             test_e, test_p = sess.run([entropy, predict], feed_dict={af:test_af, bf:test_bf, cf:test_cf, Y:Y_test})

             t_error = ROC_curve(test_p, Y_test)
             np.set_printoptions(precision=3)
             
             print("step: {}, train error: {}, test error: {} ".format(i, error, test_e/Y_test.shape[0]))
             print(t_error)
    
         if i==6000:
            [test_af, test_bf, test_cf] = build_batch_feature(X_test[0:1])
            
            _, activation =   sess.run([entropy, atom_activation], feed_dict={af:test_af, bf:test_bf, cf:test_cf, Y:Y_test[0:1]})
            
            target1 = activation[:,0,:]
            target1 = target1.reshape((1,-1))
            print(target1)
            for i in range(1):
                c = np.reshape(target1[i,:],[-1])
                d = sorted(range(len(c)), key=lambda i: c[i])
                print(d)
                

            
