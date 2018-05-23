import numpy as np
from numpy.linalg import inv
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

    def fit(self, X: Tensor) -> Tensor:
        """Build kernel matrix from a dataset."""
        raise NotImplementedError
    
    def _get_params(self, point: Tensor) -> Tuple[Tensor, Tensor]:
        """Calculate necessary parameters to augment the kernel matrix K."""
        raise NotImplementedError

    def _expand_kernel_matrix(self, K: Tensor, k: Tensor, c: Tensor) -> Tensor:
        """
        Expand base kernel matrix K right with k, then bottom with [k.T; c].
        dim(K) = (N,N)
        dim(k) = (N,n)
        dim(c) = (n,n)
        """
        K  = np.c_[K,k]
        k_ = np.c_[k.T, c]
        K  = np.r_[K, k_]
        return K

    def augment(self, new_data: Tensor) -> None:
        """Augment base kernel matrix with new noisy points."""
        new_x, new_y = new_data[:,:-1], new_data[:,-1:]
        k, c = self._get_params(new_x)
        c += (1/self.beta) * np.eye(c.shape[0])
        self.K = self._expand_kernel_matrix(self.K, k, c)       
        self.N = self.K.shape[0]
        self.X = np.r_[self.X, new_x]
        self.Y = np.r_[self.Y, new_y]

    def predict(self, x_space: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict values of new points under the current model, and their variance."""
        n_test = x_space.shape[0]
        k, c = self._get_params(x_space)
        inv_K = inv(self.K)
        yp = (k.T @ inv_K @ self.Y)[:,0]
        ys = c - k.T @ inv_K @ k
        return yp, ys
    
    def draw_samples(self, x_space: Tensor, n_samples: int = 5) -> Tensor:
        """Draw samples from the GP given its covariance matrix."""
        k, c = self._get_params(x_space)
        
        K = self._expand_kernel_matrix(self.K, k, c)
        # compute conditional on the training set
        print(c)
        nk  = self.N 
        A = inv(K)
        A_bb = A[nk:,nk:]
        A_ba = A[nk:,:nk]
        A_bb_inv = inv(A_bb)
        m_b = -A_bb_inv @ A_ba @ self.Y
        # sample from it
        print(A_bb_inv[:4,:4])
        samples = normal(m_b.T[0,:], A_bb_inv).rvs(n_samples)
        return samples


class GaussianKernel(Kernel):
    """Kernel of the form k(x, x') = \exp(-||x-x'||^2 / (2*\sigma^2))."""
    def __init__(self, data: Tensor, sigma: float = 0.1, beta: float = 0.1) -> None:
        self.X, self.Y = data[:,:-1], data[:,-1:]
        self.N, self.D = data.shape
        self.beta = beta  # precision of noise
        self.sigma = sigma 
        self.K = self.fit(self.X)
        super().__init__
    
    def fit(self, X: Tensor, noise: bool = True) -> Tensor:
        """Build kernel matrix from a dataset."""
        pairwise_dists = squareform(pdist(X, 'euclidean'))
        K = np.exp(-pairwise_dists ** 2 / (2*self.sigma ** 2))
        K = K + (1/self.beta) * np.eye(X.shape[0]) if noise else K
        return K
    
    def _get_params(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """Calculate necessary parameters to augment the kernel matrix K."""
        k = np.square(points[:,:,None] - self.X.T[None,:,:]).T
        k = np.sum(k, axis=1)
        k = np.exp(-k / (2*self.sigma ** 2))
        c = self.fit(points, noise=False)
        return k, c

class ExponentialKernel(Kernel):
    """Kernel of the form k(x, x') = \exp(-\theta |x-x'|)."""
    def __init__(self, data: Tensor, theta: float = 0.1, beta: float = 0.1) -> None:
        self.X, self.Y = data[:,:-1], data[:,-1:]
        self.N, self.D = data.shape
        self.theta = theta
        self.beta = beta  # precision of noise
        self.K = self.fit(self.X)
        super().__init__

    def fit(self, X: Tensor) -> Tensor:
        """Build kernel matrix from a dataset."""
        pairwise_dists = squareform(pdist(X, 'minkowski', p=1))
        K = np.exp(-self.theta * pairwise_dists /2) + (1/self.beta) * np.eye(self.N)
        return K

    def _get_params(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """Calculate necessary parameters to augment the kernel matrix K."""
        k = np.abs(points - self.X)
        k = np.exp(-self.theta * k / 2)
        c = np.eye(points.shape[0])
        return k, c

class ExponentialQuadraticKernel(Kernel):
    """Kernel of the form k(x, z) = \theta_0 \exp(-\theta_1/2 ||x-z||^2) + \theta_2 + \theta_3 x'z."""
    def __init__(self, data: Tensor, params: Tuple[float, ...], beta: float = 0.1) -> None:
        self.X, self.Y = data[:,:-1], data[:,-1:]
        self.N, self.D = data.shape
        self.theta_0, self.theta_1,  self.theta_2, self.theta_3 =  params
        self.beta = beta  # precision of noise
        self.K = self.fit(self.X)
        super().__init__

    def fit(self, X: Tensor) -> Tensor:
        """Build kernel matrix from a dataset."""
        #return K
        pass

    def _get_params(self, point: Tensor) -> Tuple[Tensor, Tensor]:
        """Calculate necessary parameters to augment the kernel matrix K."""
        #return k, c    
        pass