# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 19:36:45 2018

@author: Dennis
"""

import numpy as np
import time

'''Prework'''
# data initialization
np.random.seed(1)

covar = np.identity(19)
mean_class1 = np.array([1]*19)
mean_class2 = np.array([-1]*19)
x_class1 = np.random.multivariate_normal(mean_class1, covar, size = 50000)
x_class2 = np.random.multivariate_normal(mean_class2, covar, size = 50000)
x_class3 = np.random.multivariate_normal(mean_class1, covar, size = 50000)
x_class4 = np.random.multivariate_normal(mean_class2, covar, size = 50000)

data_class1 = np.c_[x_class1, np.array([1]*50000)]
data_class2 = np.c_[x_class2, np.array([-1]*50000)]
data_class3 = np.c_[x_class1, np.array([1]*50000)]
data_class4 = np.c_[x_class2, np.array([-1]*50000)]

theta0 = np.random.uniform(low = -0.5, high = 0.5, size = 19)
psi0 = np.identity(19)

train = np.r_[data_class1, data_class2]
test = np.r_[data_class3, data_class4]

'''Algorithm and function'''
# Batch Learning
def gradient_batch(data, theta):
    G = -1.71*2*0.66*(1.5*data[:,19]-1.71*np.tanh(0.66*np.dot(data[:,0:19],theta)))*(1-(np.tanh(0.66*np.dot(data[:,0:19],theta)))**2)
    g = np.dot(np.transpose(data[:,0:19]),G) 
    return g
    
def hessian_batch(data, theta):
    S = 0
    size = len(data)
    for i in range(size):
        S = S + ((1.71*0.66*(1-np.tanh(0.66*np.dot(data[i,0:19],theta))))**2)*(data[i,0:19].reshape(19,1)*data[i, 0:19])
    H = S
    return H

def Gauss_Newton(data, theta):
    
    start = time.clock()
    size = len(data)
    g_0 = gradient_batch(data, theta)
    H_0 = hessian_batch(data, theta)
    theta_old = theta
    theta_new = theta - np.dot(np.linalg.inv(H_0),g_0)
    
    while np.linalg.norm(theta_new - theta_old) >= 0.01/size:
        g_new = gradient_batch(data, theta_new)
        H_new = hessian_batch(data, theta_new)
        theta_old = theta_new
        theta_new = theta_old - np.dot(np.linalg.inv(H_new),g_new)
    
    end = time.clock()
    runtime = end - start
    
    return theta_new, runtime

# Online Learning
def gradient_online(data, theta):
    g = 2*(-1.71)*0.66*(1.5*data[19]-1.71*np.tanh(0.66*np.dot(data[0:19], theta)))*(1-(np.tanh(0.66*np.dot(data[0:19], theta)))**2)*data[0:19]
    return g

def psi_online(psi_old, theta_old, data, tau):
    psi_new = 1/(1-2./tau)*(psi_old - 1/((tau/2-1)/(1.71*0.66*(1-(np.tanh(0.66*np.dot(data,theta_old)))**2))**2+np.dot(np.dot(data,psi_old),data))*(np.dot(psi_old, data).reshape(19,1)*np.dot(psi_old, data)))
    return psi_new
#psi_new = 1/(1-2./tau)*(psi_old - 1/((tau/2-1)/(1.71*0.66*(1-(np.tanh(0.66*np.dot(data[t,0:19],theta_old)))**2))**2+np.dot(np.dot(data[t,0:19],psi_old),data[t,0:19]))*(np.dot(psi_old, data[t, 0:19]).reshape(19,1)*np.dot(psi_old, data[t, 0:19]))

def Online_Kalman(data, theta):
    
    start = time.clock()
    t = 0
    size = len(data)
    psi_old = psi0
    theta_old = theta0
    
    while True:
        t += 1
        tau = max(20, t-40)
        g = gradient_online(data[t-1,:], theta_old)
        psi_new = psi_online(psi_old, theta_old, data[t-1,0:19], tau)
        theta_new = theta_old - 1/tau*np.dot(psi_new, g)
        
        if t >= size:
            break
        else:
            psi_old = psi_new
            theta_old = theta_new
    
    end = time.clock()
    runtime = end - start
    
    return theta_new, runtime

'''Experiment'''
# data processing
theta_batch = np.zeros([30, 6, 19])
theta_online = np.zeros([30, 6, 19])
runtime_batch = np.zeros([30, 6,1])
runtime_online = np.zeros([30, 6, 1])
index = {0:1000, 1:5000, 2:10000, 3:20000, 4:50000, 5:100000}

# training
for i in range(30):
    train_p = np.random.permutation(train)
    for j in range(6):
        s_batch = Gauss_Newton(train_p[0:index[j],], theta0)
        s_online = Online_Kalman(train_p[0:index[j],], theta0)
        theta_batch[i][j] = s_batch[0]
        runtime_batch[i][j] = s_batch[1]
        theta_online[i][j] = s_online[0]
        runtime_online[i][j] = s_online[1]

theta_train_batch = np.sum(theta_batch, axis=0)/30
runtime_train_batch = np.sum(runtime_batch, axis=0)/30
theta_train_online = np.sum(theta_online, axis=0)/30
runtime_train_online = np.sum(runtime_online, axis=0)/30

t1 = theta_batch[0:28]
r1 = runtime_batch[0:28]
t2 = theta_online[0:28]
r2 = runtime_online[0:28]
theta_train_batch = np.sum(t1, axis=0)/28
runtime_train_batch = np.sum(r1, axis=0)/28
theta_train_online = np.sum(t2, axis=0)/28
runtime_train_online = np.sum(r2, axis=0)/28
    
# testing
theta_test_batch = Gauss_Newton(test, theta0)[0]
theta_test_online = Online_Kalman(test, theta0)[0]



