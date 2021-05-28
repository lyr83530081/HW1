# -*- coding: utf-8 -*-
"""
Created on Fri May 28 19:47:30 2021

@author: user
"""

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
            img=transform.resize(img, (64, 64,3))
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


class Perceptron:
    def __init__(self, n_inputs,lable_number, save_fig=False):
        self.__weights = np.random.randn(lable_number,n_inputs + 1)/10 # 1 more for bias
        self.__save_fig = save_fig
        self.__loss_train = []
        self.__loss_val = []
        self.__acc_train = []
        self.__acc_val = []
        #self.__activation = ActivationFunction(activ_func) # sign function for activation （>0 is 1,<0 is 0）

    @property
    def weights(self):
        return self.__weights[:]
    
    def weights_load(self,weights):
        self.__weights = weights

    def update_weights(self, k, X):
        #print('>> Updating') 
        self.__weights[k,:] += 0.01*X  # w += delta_w --> y * X #update weights
        #print('---------------------')

    def activation(self, y, axis=0): # sign
        #normalize
        x= y*4/abs(y).max()
        # 计算每行的最大值
        row_max = x.max(axis=axis)

        # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
        row_max=row_max.reshape(-1, 1)
        x = x - row_max

        # 计算e的指数次幂
        x_exp = np.exp(x)

        x_sum = np.sum(x_exp[0], axis=axis, keepdims=True)
        #print(x_sum)
        s = x_exp / x_sum
        #print(int(np.where(s==np.max(s))[1]))
        return s
    
    def corss_entropy_loss(self,y,target):
        #print(y.shape[1])
        B = np.zeros((1,y.shape[1]))
        B[0,target] = 1
        #print(B)
        logsoftmax = np.log(y)
        cross_loss = -np.nansum(logsoftmax*B,1)/y.shape[1]
        return cross_loss

    def check_error(self, datasets, n_iteration,lable_list):
        error = 0
        ture_t5_train = 0
        #result = False
        change_list=[]
        loss = 0
        for i in range(datasets.shape[0]):
            #print(i)
            row = datasets[:][i].tolist() #cut columns of datasets
            X, target = np.array([1] + row), lable_list[i]  # initial x[0] as 1, for bias, so that w0*x0 = w0 (aka. b)
            axis = 0
            y = self.activation(np.dot(self.__weights, X),axis=axis) # W*X = w0*x0 + w1*x1 + ... + w_n*x_n
            cross_loss = self.corss_entropy_loss(y,target)
            #print(y)
            #print(np.where(y==np.max(y)))
            flag = self.TOP_five_test(target,y)
            #et=int(np.where(y==np.max(y))[1])
            if target != int(np.where(y==np.max(y))[1]):
                change_list.append(i)

                #weight_pre[target,:] += 0.01*X         
                error += 1
                #result = X, target
            ture_t5_train += flag 
            loss+=cross_loss
        loss = loss/datasets.shape[0]
        acc_t = (len(lable_list)-error)/len(lable_list)*100
        self.__loss_train.append(loss)
        self.__acc_train.append(acc_t)
        print('Iteration #{}: accuracy_train = {}%, accuracy_t5 = {}%'.format(n_iteration, acc_t,
                                                           100*ture_t5_train/len(lable_list)))
        return change_list
    
    
    def TOP_five_test(self,lable,outputs_val):
        outputs = 1*outputs_val[0]
        flag=0
        for n in range(5):
            #print(int(np.where(outputs==np.max(outputs))[1]))
            index=int(np.where(outputs==np.max(outputs))[0])
            if lable == index:
                flag = 1
            outputs[index] = -1
        return flag 
    
    def validation(self, val_data,n_iteration, lable_list_val):
        error = 0
        ture_t5=0
        loss = 0
        for i in range(val_data.shape[0]):
            row = val_data[i,:].tolist() #cut columns of datasets
            X, target = np.array([1] + row), lable_list_val[i]  # initial x[0] as 1, for bias, so that w0*x0 = w0 (aka. b)
            axis = 0
            y = self.activation(np.dot(self.__weights, X),axis=axis) # W*X = w0*x0 + w1*x1 + ... + w_n*x_n
            cross_loss = self.corss_entropy_loss(y,target)
            flag = self.TOP_five_test(target,y)
            #print (target, int(np.where(y==np.max(y))[1]))
            if target != int(np.where(y==np.max(y))[1]):
                error += 1
            ture_t5+=flag
            loss+=cross_loss
        loss = loss/val_data.shape[0]
        self.__loss_train.append(loss)   
        accuracy_val=(len(lable_list_val)-error)/len(lable_list_val)*100
        print('Iteration #{}: test_validation = {}%, test_t5_val = {}%'.format(n_iteration,accuracy_val,
                                                                          100*ture_t5/len(lable_list_val)))
        self.__acc_val.append(accuracy_val)
        #print(100*accuracy_val, 100*ture_t5/len(lable_list_val))
        return error
    
    def train(self, datasets, val_data,lable_list, lable_list_val):
        iterations = 10
        
        n_weights = self.__weights.shape[1]
        if datasets.shape[1] != n_weights-1: 
            raise Exception("Wrong inputs of training!")

        
        for i in range(iterations): 
            print('---------------------------') 
            n_iteration = i+1
            change_list = self.check_error(datasets, n_iteration,lable_list)
            error_validation = self.validation(val_data,n_iteration, lable_list_val)
            for k in change_list:
                weight_change,target=np.array([1] + datasets[:][k].tolist()),lable_list[k]
                self.update_weights(target,weight_change)  
            np.save(r"checkpoint/%s.npy" % str(i), self.weights)
        return self.__acc_train,self.__acc_val
    
weights = np.load(r"checkpoint/9.npy")
test_data, lable_list_test = feature_extrat('test.txt')
lable_number=50
n_inputs = 256*3
# build model
myPerceptron = Perceptron(n_inputs=n_inputs, lable_number=lable_number, save_fig=False)
myPerceptron.weights_load(weights)
myPerceptron.validation(test_data, 0, lable_list_test)

