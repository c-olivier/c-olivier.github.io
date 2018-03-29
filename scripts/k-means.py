#!/usr/bin/env python3

#====================================================================================================
# title           : k-means.py
# description     : Implementation of K-means for K=2.
# author          : Charles Olivier (https://c-olivier.github.io/)
# date            : 29.03.2018
# usage           : chmod +x k-means.py; ./k-means.py
# notes           : The algorithm is sensitive to the choice of initial conditions.
# python_version  : 3.6.4
#====================================================================================================


import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

from scipy.stats import multivariate_normal as normal
from scipy.stats import norm

# generate datasets
n = 50
x1 = normal([-1,-1], 0.2*np.eye(2)).rvs(n)
x2 = normal([+1,+1], 0.2*np.eye(2)).rvs(n)
x = np.concatenate((x1,x2), axis=0)
np.random.shuffle(x)

mu, cov = np.mean(x, axis=0), np.cov(x.T)
x = (np.linalg.inv(np.linalg.cholesky(cov)) @ x.T).T

#plt.plot(x[:,0], x[:,1], 'o', color='green');

steps = 12
#mu1, mu2 = np.array([0.1,-0.3]), np.array([1,-1])
mu1, mu2 = np.random.normal(size=2), np.random.normal(size=2) # random initialisation of means
plt.figure(figsize=(8,3*steps/2.5))
plt.suptitle('K-means', y=1)
for i in range(steps):
    # E step: calculate responsibilities
    rn1 = np.sum(np.square(x-mu1), axis=1, keepdims=True)
    rn2 = np.sum(np.square(x-mu2), axis=1, keepdims=True)
    g1, g2 = np.meshgrid(np.arange(-2.5,+2.5+0.06, 0.05),\
                         np.arange(-2.5,+2.5+0.06, 0.05))
    g = np.c_[g1.ravel(), g2.ravel()]
    g_pred = (np.sum(np.square(g-mu1), axis=1, keepdims=True)<np.sum(np.square(g-mu2), axis=1, keepdims=True)).astype(int).reshape(*g1.shape)
    rn_ = np.concatenate((rn2, rn1), axis=1)
    rn = np.zeros_like(rn_)
    rn[np.arange(rn_.shape[0]), rn_.argmax(1)] = 1
    N = np.sum(rn, axis=0)
    # M step: optimise parameters
    x1 = x[rn[:,0]==1]
    x2 = x[rn[:,1]==1]
    mu1 = np.sum(x1, axis=0)/N[0]
    mu2 = np.sum(x2, axis=0)/N[1]
    J = np.sum(np.square(x1 - mu1)) + np.sum(np.square(x2-mu2))
    plt.subplot(int(steps/2+0.5),4,i+1)
    plt.title('Step {}'.format(i+1))
    plt.contourf(g1, g2, g_pred)
    plt.plot(x1[:,0], x1[:,1], 'bo')
    plt.plot(x2[:,0], x2[:,1], 'go')
    plt.plot(mu1[0], mu1[1], 'r+', ms='20')
    plt.plot(mu2[0], mu2[1], 'r+', ms='20', label='J={:.3f}'.format(J))
    plt.axis('scaled')
    plt.tight_layout()
    #plt.axis([-2, 2, -2, 2])
plt.show();