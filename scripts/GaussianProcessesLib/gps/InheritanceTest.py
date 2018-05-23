import numpy as np
from numpy.linalg import inv
from tensor import Tensor
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky, cho_solve
from typing import Tuple, Any, Iterable


import matplotlib.pyplot as plt


class Kernel:
    def __init__(self, X: Tensor, Y: Tensor, beta: float = 0.1) -> None:
        self.X = X
        self.Y = Y
        self.beta = beta
        self.K = self.fit(X)
        self.N, self.D = X.shape
        
    @classmethod
    def from_dataset(cls: Any, data: Tensor, beta: float = 0.1, **kwargs: float) -> Any:
        X, Y = data[:,:-1], data[:,-1:]
        return cls(X, Y, beta, **kwargs)

    def fit(self, X: Tensor, noise: bool = True) -> Tensor:
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
        print(K.shape, k.shape, c.shape)
        K  = np.concatenate((K,k), axis=1)
        k_ = np.concatenate((k.T, c), axis=1)
        K  = np.concatenate((K, k_), axis=0)
        return K

    def augment(self, new_data: Tensor) -> None:
        """Augment base kernel matrix with new noisy points."""
        new_x, new_y = new_data[:,:-1], new_data[:,-1:]
        k, c = self._get_params(new_x)
        c += (1/self.beta) * np.eye(c.shape[0])
        self.K = self._expand_kernel_matrix(self.K, k, c)       
        self.N = self.K.shape[0]
        self.X = np.concatenate((self.X, new_x), axis=0)
        self.Y = np.concatenate((self.Y, new_y), axis=0)

    def predict(self, x_space: Tensor, return_std: bool = False, return_cov: bool = False) -> Iterable[Tensor]:
        """Predict values of new points under the current model, and their variance."""
        k, c = self._get_params(x_space)
        L = cholesky(self.K, lower=True)
        alpha = cho_solve((L, True), self.Y)
        predicted_mean = (k.T @ alpha)[:,0]

        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")
        
        if return_cov or return_std:
            v = cho_solve((L, True), k)
            predicted_cov = c - k.T @ v

        if return_cov: return (predicted_mean, predicted_cov)
        
        elif return_std: return (predicted_mean, np.sqrt(np.diag(predicted_cov)))

        return predicted_mean
    
    def draw_samples(self, x_space: Tensor, n_samples: int = 5) -> Tensor:
        """Draw samples from the GP given its covariance matrix."""
        gauss_params = self.predict(x_space, return_cov=True)
        #print(gauss_params[1])
        #print(multivariate_normal(*gauss_params).rvs(n_samples).shape)
        return multivariate_normal(*gauss_params).rvs(n_samples)

        k, c = self._get_params(x_space)
        
        K = self._expand_kernel_matrix(self.K, k, c)
        # compute conditional on the training set
        nk  = self.N 
        A = inv(K)
        A_bb = A[nk:,nk:]
        A_ba = A[nk:,:nk]
        A_bb_inv = inv(A_bb)
        m_b = -A_bb_inv @ A_ba @ self.Y
        # sample from it
        print(A_bb_inv[:4,:4])
        A_bb_inv += 0.1*np.eye(A_bb_inv.shape[0])
        samples = multivariate_normal(m_b.T[0,:], A_bb_inv).rvs(n_samples)
        return samples


class GaussianKernel(Kernel):
    def __init__(self, X: Tensor, Y: Tensor, beta: float = 10, sigma: float = 1) -> None:
        self.sigma = sigma
        super().__init__(X, Y, beta)

    def fit(self, X: Tensor, noise: bool = True) -> Tensor:
        """Build kernel matrix from a dataset."""
        pairwise_dists = squareform(pdist(X, 'euclidean'))
        K = np.exp(-pairwise_dists ** 2 / (2*self.sigma ** 2))
        if noise:
            return K + (1/self.beta) * np.eye(X.shape[0])
        return K

    def _get_params(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """Calculate necessary parameters to augment the kernel matrix K."""
        k = np.square(points[:,:,None] - self.X.T[None,:,:]).T
        k = np.sum(k, axis=1)
        k = np.exp(-k / (2*self.sigma ** 2))
        c = self.fit(points, noise=False)
        return k, c


N, D = 10, 1
x = np.linspace(1,10, N) + 0.1*(2*np.random.random(N)-1)
x = x[:,None]
y = np.cos(x) + (2*np.random.random((N, 1))-1)

data = np.concatenate((x,y), axis=1)


kernel = GaussianKernel.from_dataset(data, sigma=1, beta=100)





n_test = 10
z = np.linspace(1,10,n_test)
z = z[:,None]

# draw_sample test
p = kernel.draw_samples(z, 5)
print(p.shape)

# fit test

pred_mean, pred_std = kernel.predict(z, return_std=True)

plt.figure()
z = z[:,0]
plt.plot(z, pred_mean, label=r'$Predicted \pm 1\sigma$')
plt.plot(x, y, 'o', label='Training Points')
plt.plot(z, np.cos(z), label='True Distribution')
plt.fill_between(z, pred_mean+pred_std, pred_mean-pred_std, alpha=.2)
for sample in p:
    plt.plot(z, sample, 'k--')

plt.legend(loc=0)
plt.show()
