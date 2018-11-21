#Use a multioutput random forest

import numpy as np
from mol_feature import load_data, circular_fps
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

input_dim = 1024
output_dim = 12

def data_statistics(Y_test):
    tvalid = np.logical_not(np.isnan(Y_test))

    total_pos = np.where(tvalid, Y_test, 0)
    print("Total: ", np.sum(tvalid, axis=0))
    total_neg = np.where(tvalid, (1-Y_test), 0)
    rate = np.sum(total_pos,axis=0) / ( np.sum(total_pos,axis=0) +  np.sum(total_neg,axis=0))
    print("Pos rate: ------", rate)
    print("weights: ", 0.5/rate)
    
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
                
[smiles, labels] = load_data('data.csv') 
X_train = circular_fps(smiles[0:7000])
Y_train = labels[0:7000]
Y_train = np.nan_to_num(Y_train)
X_test = circular_fps(smiles[7000:])
Y_test = labels[7000:,:]
Y_test = np.nan_to_num(Y_test)

#balance_weight = [{0:1,1:10.5}, {0:1,1:15.5},  {0:1,1:4.5},  {0:1,1:9.5},  {0:1,1:3.5}, {0:1,1:8.5}, {0:1,1:18.5}, {0:1,1:3.5}, {0:1,1:14.5}, {0:1,1:8.5}, {0:1,1:3.5}, {0:1,1:9.5}]
forest = RandomForestClassifier(n_estimators=input_dim, random_state=1, min_samples_split=10, class_weight="balanced")
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest.fit(X_train, Y_train)

test_predict = multi_target_forest.predict(X_test)
rt = accuracy(test_predict, Y_test, 0.5)
print(rt[0])
print(rt[1])
