#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:26:12 2020

@author: josephharvey
"""
import numpy as np
from numba import jit

a = 10
b = 28
c = 8/3
h = 0.01
T = 100

@jit(nopython = True, fastmath = True)
def fx(x,y):
    return -a*x + a*y
    
@jit(nopython = True, fastmath = True)
def fy(x,y,z):
    return b*x - y - x*z

@jit(nopython = True, fastmath = True)
def fz(x,y,z):
    return -c*z + x*y

#@jit(nopython = True, fastmath = True)
def rungekutta(x0 = 1,y0 = 1,z0 = 1, h = 0.01, T = 100):
    xarr = np.array([x0])
    yarr = np.array([y0])
    zarr = np.array([z0])
    
    #loops from t = 0 to T 
    for i in range(0, int(T/h)):
        
        k1x = fx(xarr[i], yarr[i])
        k1y = fy(xarr[i], yarr[i], zarr[i])
        k1z = fz(xarr[i], yarr[i], zarr[i])
        
        k2x = fx(xarr[i] + h*k1x/2, yarr[i] + h*k1y/2)
        k2y = fy(xarr[i] + h*k1x/2, yarr[i] + h*k1y/2, zarr[i] + h*k1z/2)
        k2z = fz(xarr[i] + h*k1x/2, yarr[i] + h*k1y/2, zarr[i] + h*k1z/2)
        
        k3x = fx(xarr[i] + h*k2x/2, yarr[i] + h*k2y/2)
        k3y = fy(xarr[i] + h*k2x/2, yarr[i] + h*k2y/2, zarr[i] + h*k2z/2)
        k3z = fz(xarr[i] + h*k2x/2, yarr[i] + h*k2y/2, zarr[i] + h*k2z/2)
        
        k4x = fx(xarr[i] + h*k3x, yarr[i] + h*k3y)
        k4y = fy(xarr[i] + h*k3x, yarr[i] + h*k3y, zarr[i] + h*k3z)
        k4z = fz(xarr[i] + h*k3x, yarr[i] + h*k3y, zarr[i] + h*k3z)
        
        xarr = np.append(xarr, (xarr[i] + 1/6*h*(k1x + 2*k2x + 2*k3x + k4x)))
        yarr = np.append(yarr, (yarr[i] + 1/6*h*(k1y + 2*k2y + 2*k3y + k4y)))
        zarr = np.append(zarr, (zarr[i] + 1/6*h*(k1z + 2*k2z + 2*k3z + k4z)))
        
    
    return np.stack([xarr,yarr,zarr])

#u = rungekutta(1,1,1)

#time = np.array(range(0,2201))

#plt.plot(time, u[0])

#u_longterm = u[:,200:]
#time = np.array(range(0, u_longterm[0].size))
#plt.plot(u_longterm[2], u_longterm[0])

#u_sample = u[:, 0::10]
#print(u_sample.shape)
#time = np.array(range(0, u_sample[0].size))
#plt.plot(time, u_sample[0])

