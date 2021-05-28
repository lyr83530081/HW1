# -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:53:34 2021

@author: user
"""

from sklearn.model_selection import train_test_split
from sklearn import metrics
from  sklearn.datasets  import  make_hastie_10_2
from xgboost.sklearn import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import transform



def feature_extrat(data_path):
    with open(data_path, 'r') as f:
        data = f.readlines()  #txt中所有字符串读入data
        lable_list = [0]*len(data)
        k=0
        X = np.array([[]]*768).T
        for line in data:
            img_path = line.split()        #将单个数据分隔开存好
            lable_list[k] = int(img_path[1])
            k+=1
            if k % 2000 == 0:
                print(k)
    
            img = cv2.imread(img_path[0])
            #plt.imshow(img)
            #plt.show
            img=transform.resize(img, (227, 227,3))
            b = img[:,:,0]*255
            g = img[:,:,1]*255
            r = img[:,:,2]*255
            b = b.astype(np.uint8)
            g = g.astype(np.uint8)
            r = r.astype(np.uint8)
            feature_b = cv2.calcHist([b],[0],None,[256],[0,256]).reshape(1,-1)
            feature_g = cv2.calcHist([g],[0],None,[256],[0,256]).reshape(1,-1)
            feature_r = cv2.calcHist([r],[0],None,[256],[0,256]).reshape(1,-1)
            feature = np.hstack((feature_b,feature_g,feature_r))
            feature = feature/feature.sum()
            #print(k)
            X = np.append(X,feature,axis=0)
            #print(X.shape)
    print(k)
    return X, lable_list

def TOP_five_test(lable,outputs_val):
    outputs = 1*outputs_val[:]
    flag=0
    for n in range(5):
        #print(int(np.where(outputs==np.max(outputs))[1]))
        index=int(np.where(outputs==np.max(outputs))[1])
        if lable == index:
            flag = 1
        outputs[0][index] = -1
    return flag 


X=np.load('feature_train.npy')
lable_list=np.load('lable.npy')
#print(X[0:10,1:10])
val_data, lable_list_val = feature_extrat('val.txt')
clf = XGBClassifier(

        n_estimators=100,

        learning_rate= 0.3, 

        max_depth=6, 

        subsample=1, 
        use_label_encoder = False,
        gamma=0, 
        reg_lambda=1,  
        max_delta_step=0,
        colsample_bytree=1, 
        min_child_weight=1, 
        seed=1000 
)

clf.fit(X,lable_list,eval_metric='auc')
y_pred=clf.predict(val_data)
y_prob=clf.predict_proba(val_data)

val_label = np.array(lable_list_val)
true_num = 0
for i in range(len(val_label)):
    if val_label[i] == y_pred[i]:
        true_num +=1
        
accuracy = true_num/len(val_label)

#print(y_prob[0])
true_t5 = 0
for i in range(len(val_label)):
    outputs = 1*y_prob[i]
    flag=0
    for n in range(5):
        #print(int(np.where(outputs==np.max(outputs))[0]))
        index=int(np.where(outputs==np.max(outputs))[0])
        if val_label[i] == index:
            flag = 1
        outputs[index] = -1.0
    true_t5+=flag

acc_t5 = true_t5/len(val_label)