#!/usr/bin/env python3

#========================================================================================================================
# title           : mixture_of_gaussians_em.py
# description     : Implementation of the Expectation Maximisation (EM) algorithm for a mixture of K Gaussians.
# author          : Charles Olivier (https://c-olivier.github.io/)
# date            : 29.03.2018
# usage           : chmod +x mixture_of_gaussians_em.py; ./mixture_of_gaussians_em.py
# notes           : The algorithm is sensitive to the choice of initial conditions. K (number of clusters) can be changed.
# python_version  : 3.6.4
#=========================================================================================================================


import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

from scipy.stats import multivariate_normal as normal
from scipy.stats import norm

from random import shuffle
cor = ['#fad390', '#f6b93b', '#fa983a', '#e58e26', '#f8c291', '#e55039', '#eb2f06', '#b71540']
cog = ['#b8e994', '#78e08f', '#38ada9', '#079992']
cob = ['#6a89cc', '#4a69bd', '#1e3799', '#0c2461', '#82ccdd', '#60a3bc', '#3c6382', '#0a3d62']
mixed = cor[:3]+cog[:3]+cob[:3]
shuffle(mixed)

# generate datasets
n = 50
x1 = normal([-1,-1], 0.2*np.eye(2)).rvs(n)
x2 = normal([+1,+1], 0.8*np.eye(2)).rvs(n)
x = np.concatenate((x1,x2), axis=0)
np.random.shuffle(x)

mu, cov = np.mean(x, axis=0), np.cov(x.T)
x = (np.linalg.inv(np.linalg.cholesky(cov)) @ x.T).T

K = 3 # number of clusters
pi  = np.ones(shape=(1,K))/K
mu  = np.random.uniform(size=(2,K))
cov = np.c_[[0.01*np.eye(2) for i in range(K)]]
gam = np.zeros((x.shape[0], K)) # responsibilities
steps = 10
plot_every = 1
plot_count = 1
log_likelihood = [0]

plt.figure(figsize=(8,3*steps/4.5/plot_every))
plt.suptitle('Mixture of Gaussians (EM)', y=1.02)
for i in range(steps):
    # Expectation
    for k in range(K):
         gam[:,k] = normal(mu[:,k], cov[k,:]).pdf(x) * pi[0][k]
    LL = np.sum(np.log(np.sum(gam, axis=1))) # log likelihood
    log_likelihood.append(LL)
    #if not (i % 10): print('Step {}: LL = {:.3f}'.format(i, LL))
    gam /= np.sum(gam, axis=1)[:,None]
    # Maximisation
    Nk = np.sum(gam, axis=0)
    for k in range(K):
        mu[:,k]  = 1/Nk[k] * np.sum(gam[:,k,None] * x, axis=0) # mean
        cov[k,:] = 1/Nk[k] * (np.sqrt(gam[:,k,None]) * (x - mu[:,k])).T @ (np.sqrt(gam[:,k,None]) * (x - mu[:,k])) # covariance
        pi[0,k]  = Nk[k] / x.shape[0] # coefficients
    
    # Visualisation
    if i % plot_every == 0:
        plt.subplot(int(steps/plot_every/4+0.5),4,plot_count)
        plt.title('Step {}'.format(i+1))
        # Plot
        g1, g2 = np.meshgrid(np.arange(np.min(x[:,0]),np.max(x[:,0])+0.06, 0.05),\
                             np.arange(np.min(x[:,1]),np.max(x[:,1])+0.06, 0.05))
        g = np.c_[g1.ravel(), g2.ravel()]
        g_pred = np.sum(pi * np.concatenate([normal(mu[:,k], cov[k,:]).pdf(g)[:,None] for k in range(K)], axis=1), axis=1).reshape(*g1.shape)

        for k in range(K):
            plt.plot(mu[0,k], mu[1,k], 'r+', ms='20')
        plt.contourf(g1, g2, g_pred)
        plt.scatter(x[:,0], x[:,1], c=[mixed[i] for i in np.argmax(gam, axis=1)])
        plt.axis([np.min(x[:,0]), np.max(x[:,0]), np.min(x[:,1]), np.max(x[:,1])])
        plot_count += 1
    # Early stopping if LL converges 
    #try: np.testing.assert_almost_equal(log_likelihood[-1], log_likelihood[-2], decimal=5)
    #except AssertionError: continue
    #else: break
plt.tight_layout()
plt.show()


plt.figure()
plt.semilogy(np.arange(1,len(log_likelihood),1), -np.array(log_likelihood)[1:])
plt.xlabel('Step')
plt.title('-Log Likelihood')
plt.show();