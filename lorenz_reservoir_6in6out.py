#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:57:49 2020

@author: josephharvey
"""

from lorenzrungekutta import rungekutta
import numpy as np
#from sklearn.linear_model import Ridge
from scipy import sparse
from scipy.linalg import solve
from scipy.sparse.linalg import eigs
from matplotlib import pyplot as plt

np.random.seed()

#why is lorenz signal passing 0?

#50 seems to be unstable boundary
class Reservoir:
    def __init__(self, rk, rsvr_size = 300, spectral_radius = 0.6, input_weight = 1):
        self.rsvr_size = rsvr_size
        
        #get spectral radius < 1
        #gets row density = 0.03333
        unnormalized_W = (np.random.rand(rsvr_size,rsvr_size)*2 - 1)
        for i in range(unnormalized_W[:,0].size):
            for j in range(unnormalized_W[0].size):
                if np.random.rand(1) > 10/rsvr_size:
                    unnormalized_W[i][j] = 0
    
        max_eig = eigs(unnormalized_W, k = 1, return_eigenvectors = False)
        
        self.W = sparse.csr_matrix(spectral_radius/np.abs(max_eig)*unnormalized_W)
        
        const_conn = int(rsvr_size*0.1)
        signal_conn = int((rsvr_size-const_conn)/6)
        
        
        Win = np.zeros((rsvr_size, 7))
        #constant
        Win[:const_conn, 0] = (np.random.rand(Win[:const_conn, 0].size)*2 - 1)*input_weight
        #x
        Win[const_conn: const_conn + signal_conn, 1] = (np.random.rand(Win[const_conn: const_conn + signal_conn, 1].size)*2 - 1)*input_weight
        #y
        Win[const_conn + signal_conn:const_conn + 2*signal_conn, 2] = (np.random.rand(Win[const_conn + signal_conn:const_conn + 2*signal_conn, 2].size)*2 - 1)*input_weight
        #z
        Win[const_conn + 2*signal_conn: const_conn + 3*signal_conn, 3] = (np.random.rand(Win[const_conn + 2*signal_conn: const_conn + 3*signal_conn, 3].size)*2 - 1)*input_weight
        #x^2
        Win[const_conn + 3*signal_conn:const_conn + 4*signal_conn, 4] = (np.random.rand(Win[const_conn + 3*signal_conn:const_conn + 4*signal_conn, 4].size)*2 - 1)*input_weight
        #y^2
        Win[const_conn + 4*signal_conn:const_conn + 5*signal_conn, 5] = (np.random.rand(Win[const_conn + 4*signal_conn:const_conn + 5*signal_conn, 5].size)*2 - 1)*input_weight
        #z^2
        Win[const_conn + 5*signal_conn:, 6] = (np.random.rand(Win[const_conn + 5*signal_conn:, 6].size)*2 - 1)*input_weight
        
        
        self.Win = sparse.csr_matrix(Win)
        self.X = (np.random.rand(rsvr_size, rk.train_length+2)*2 - 1)
        self.Wout = np.array([])
        
class RungeKutta:
    def __init__(self, x0 = 2,y0 = 2,z0 = 23, h = 0.01, T = 300, ttsplit = 5000, noise_scaling = 0):
        u_arr = rungekutta(x0,y0,z0,h,T)[:, ::5]
        self.train_length = ttsplit
        
        u_arr[0] = (u_arr[0] - 0)/7.929788629895004
        u_arr[1] = (u_arr[1] - 0)/8.9932616136662
        u_arr[2] = (u_arr[2] - 23.596294463016896)/8.575917849311919
        
        self.u_arr_train = u_arr[:, :ttsplit+1]
        #size 5001
        
        #noisy training array
        #switch to gaussian 
        noise = np.random.randn(self.u_arr_train[:,0].size, self.u_arr_train[0,:].size)*noise_scaling 
        self.u_arr_train_noise = self.u_arr_train + noise
        
        #plt.plot(self.u_arr_train_noise[0, :500])
        
        #u[5000], the 5001st element, is the last in u_arr_train and the first in u_arr_test
        self.u_arr_test = u_arr[:, ttsplit:]
        #size 1001
  
#takes a reservoir object res along with initial conditions
def getX(res, rk,x0 = 1,y0 = 1,z0 = 1, noise = False):
    
    if noise:
        u_training = rk.u_arr_train_noise
    else:
        u_training = rk.u_arr_train
    
    #loops through every timestep
    for i in range(0, u_training[0].size):
        u = np.concatenate((np.array([1]), u_training[:,i], transform(u_training[:,i]))).reshape(7,1)
        
        x = res.X[:,i].reshape(res.rsvr_size,1)
        x_update = np.tanh(np.add(res.Win.dot(u), res.W.dot(x)))
        
        res.X[:,i+1] = x_update.reshape(1,res.rsvr_size)    
    
    return res.X
    
def trainRRM(res, rk):
    print("Training... ")

    alph = 10**-4
    #rrm = Ridge(alpha = alph, solver = 'cholesky')
    
    #train on 10 small training sets with different noise - minimize error over all
    #save the state of the reservoir for noisy datasets
    #also try - train on signal^2 or other function (get more info than just 3 vars) - no noise
    
    Y_train = np.concatenate((rk.u_arr_train[:, 301:], transform(rk.u_arr_train[:, 301:])), axis = 0)
    
    X = getX(res, rk, noise = True)[:, 301:(res.X[0].size - 1)]
    X_train = np.concatenate((np.ones((1, rk.u_arr_train[0].size-301)), X, rk.u_arr_train[:, 300:(rk.u_arr_train[0].size - 1)], transform(rk.u_arr_train[:, 300:(rk.u_arr_train[0].size - 1)])), axis = 0)
    #X_train = np.copy(X)
    
    idenmat = np.identity(res.rsvr_size+7)*alph
    data_trstates = np.matmul(Y_train, np.transpose(X_train))
    states_trstates = np.matmul(X_train,np.transpose(X_train))
    res.Wout = np.transpose(solve(np.transpose(states_trstates + idenmat),np.transpose(data_trstates)))
    
    print("Training complete ")
    #Y_train = Y_train.transpose()
    #X_train = X.transpose()
    
    #tweak regression param? use 10^-4, 10^-6
    #test Ridge() in simpler context
    #rrm.fit(X_train,Y_train)
    #res.Wout = rrm.coef_
    return
    
def predict(res, x0 = 0, y0 = 0, z0 = 0, steps = 1000):
    Y = np.empty((6, steps + 1))
    X = np.empty((res.rsvr_size, steps + 1))
    
    Y[:3,0] = np.array([x0,y0,z0]).reshape(1,3)
    Y[3:,0] = transform(np.array([x0,y0,z0]).reshape(1,3))
    X[:,0] = res.X[:,-2]

    
    for i in range(0, steps):
        y_in = np.concatenate((np.array([1]), Y[:,i]), axis = 0).reshape(7,1) 
        x_prev = X[:,i].reshape(res.rsvr_size,1)
        
        x_current = np.tanh(np.add(res.Win.dot(y_in), res.W.dot(x_prev)))
        X[:,i+1] = x_current.reshape(1,res.rsvr_size)
        #X = np.concatenate((X, x_current), axis = 1)
        
        y_out = np.matmul(res.Wout, np.concatenate((np.array([[1]]), x_current, Y[:,i].reshape(6,1)), axis = 0))
        #y_out = np.matmul(res.Wout, x_current)
        Y[:,i+1] = y_out.reshape(1, 6) 
        

    return Y

def test(res, num_tests = 10, rkTime = 200, split = 2000, showPlots = True):
    valid_time = np.array([])
    means = np.zeros(num_tests)
    variances = np.zeros(num_tests)
    
    for i in range(num_tests):
        ic = np.random.rand(3)*2-1
        rktest = RungeKutta(x0 = ic[0], y0 = ic[1], z0 = 30*ic[2], T = rkTime, ttsplit = split)
        res.X = (np.zeros((res.rsvr_size, split+2))*2 - 1)
        
        #sets res.X
        getX(res, rktest)
        
        pred = predict(res, x0 = rktest.u_arr_test[0,0], y0 = rktest.u_arr_test[1,0], z0 = rktest.u_arr_test[2,0], steps = (rkTime*20-split))
        
        for j in range(0, pred[0].size):
            if np.abs(pred[0, j] - rktest.u_arr_test[0, j]) > 1.5:
                valid_time = np.append(valid_time, j)
                print("Test " + str(i) + " valid time: " + str(j))
                break
        
        print("Mean: " + str(np.mean(pred[0])))
        print("Variance: " + str(np.var(pred[0])))
        
        means[i] = np.mean(pred[0])
        variances[i] = np.var(pred[0])
        
        if showPlots:
            plt.figure()
            plt.plot(pred[0])
            plt.plot(rktest.u_arr_test[0])
    
    if showPlots:
        plt.show()
    
    print("Avg. valid time steps: " + str(np.mean(valid_time)))
    print("Std. valid time steps: " + str(np.std(valid_time)))
    print("Avg. of x dim: " + str(np.mean(means)))
    print("Var. of x dim: " + str(np.mean(variances)))
    return np.mean(valid_time)

def transform(x):
    return np.tanh(x)*0.0001 

#fidelity metric
#valid time - consider three inputs and 6 inputs
#try at a few res. sizes

#stability criterion, comparing no noise, 3 inputs, 6 inputs, noise: 
#eyeball test
#mean and variance of all datapoints
#intervals between sign changes 
#compute correlation function 

train_time = 300    
rk = RungeKutta(T = train_time, ttsplit = train_time*20, noise_scaling = 0.000) #ttsplit = train_time*20

results = np.array([])
num_res = 10
for i in range (num_res):
    print("Reservoir " + str(i+1) + " of " + str(num_res))
    res = Reservoir(rk, rsvr_size = 50, spectral_radius = 0.5, input_weight = 1)
    trainRRM(res, rk) 
    results = np.append(results, test(res, 5, rkTime = 200, showPlots = True))
    print(results.size)
print("Average valid time for " + str(num_res) + " reservoirs at " + str(train_time) + ": " + str(np.mean(results)))
print(results) 

