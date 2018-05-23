#!/usr/local/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from tensor import Tensor
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal as normal
from typing import Tuple


class Kernel:
    def __init__(self, K: Tensor, N: int, beta: float, X: Tensor, Y: Tensor) -> None:
        self.K = K
        self.N = N
        self.X, self.Y = X, Y
        self.beta = beta

    def fit(self) -> Tensor:
        """Build kernel matrix from the training set."""
        raise NotImplementedError
    
    def _get_params(self, point: Tensor) -> Tuple[Tensor, float]:
        """Calculate necessary parameters to augment the kernel matrix K."""
        raise NotImplementedError
    
    def add_datum(self, point: Tensor) -> Tensor:
        """Augment base kernel matrix with a new data point."""
        k, c = self._get_params(point)
        self.K = np.concatenate((self.K, k), axis=1)
        k_ = np.zeros((1, self.N+1))
        k_[0,:self.N] = k.T
        k_[0,-1] = c
        self.K = np.concatenate((self.K, k_), axis=0)
        self.N = self.K.shape[0]
        return self.K

    def predict(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict values of new points under the current model, and their variance."""
        print(points.shape)
        k, c = self._get_params(points)
        print('!!', k.shape)
        inv_K = np.linalg.inv(self.K)
        return k.T @  inv_K @ self.Y, c - k.T @ inv_K @ k
    
    def draw_samples(self, n: int = 10) -> Tensor:
        """Draw samples from the GP given its covariance matrix."""
        return normal(np.zeros(self.N), self.K).rvs(n)


class GaussianKernel(Kernel):
    def __init__(self, data: Tensor, sigma: float = 0.1, beta: float = 0.1) -> None:
        self.X, self.Y = data[:,:-1], data[:,-1:]
        self.N, self.D = data.shape
        self.beta = beta  # precision of noise
        self.sigma = sigma 
        self.K = self.fit()
        super().__init__
    
    def fit(self) -> Tensor:
        """Build kernel matrix from the training set."""
        pairwise_dists = squareform(pdist(self.X, 'euclidean'))
        K = np.exp(-pairwise_dists ** 2 / (2*self.sigma ** 2)) + (1/self.beta) * np.eye(self.N)
        return K
    
    def _get_params(self, point: Tensor) -> Tuple[Tensor, float]:
        """Augment base kernel matrix with a new data point."""
        k = np.sum(np.square(point - self.X), axis=1, keepdims=True)
        k = np.exp(-k / (2*self.sigma ** 2))
        c = 1 + (1/self.beta)
        return k, c
        
        


N, D = 50, 1
x = np.linspace(1,10, N)
x = x[:,None]
y = x + np.random.random((N, 1))
#data = np.random.random((N, D+1)) # [X; y]

data = np.concatenate((x,y), axis=1)
kernel = GaussianKernel(data, sigma=0.5, beta=100)

#print(kernel.K[:4,:4])

new = np.random.random((1, D)) + 4
#K = kernel.add_datum(new)
x = np.linspace(1,10, 20)
x = x[:,None]
p = kernel.predict(x)


# n=4
# s = kernel.draw_samples(n)

# plt.figure(figsize=(8,3))
# plt.subplot(121)
# plt.imshow(kernel.K)


# plt.subplot(122)
# for i in range(n):
#     plt.plot(x[:,0], s[i,:])
# plt.show()
# print(s.shape)

#print(kernel.K.shape)
