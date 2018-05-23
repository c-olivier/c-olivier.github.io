#!/usr/local/bin/python3
from kernels import *
import matplotlib.pyplot as plt

def main():
    N, D = 3, 1
    x = np.linspace(1,10, N) + 0.1*(2*np.random.random(N)-1)
    x = x[:,None]
    y = np.cos(x) + (2*np.random.random((N, 1))-1)


    data = np.concatenate((x,y), axis=1)
    kernel = GaussianKernel(data, sigma=0.8, beta=10000)
    #kernel = ExponentialKernel(data, theta=1, beta=5)

    # augmentation test
    #kernel.augment(data+0.1)
    #print('Augmented Kernel shape', kernel.K.shape)
    
    n_test = 100
    z = np.linspace(0,10,n_test)
    z = z[:,None]

    # draw_sample test
    #p = kernel.draw_samples(z, 10)
    #print(p.shape)

    # fit test
    
    yp, ys = kernel.predict(z)
    ys = np.diag(ys)
    ys = 1*np.sqrt(ys) # 68% CI

    plt.figure()
    z = z[:,0]
    plt.plot(z, yp, label=r'$Predicted \pm 1\sigma$')
    plt.plot(x, y, 'o', label='Training Points')
    plt.plot(z, np.cos(z), label='True Distribution')
    plt.fill_between(z, yp+ys, yp-ys, alpha=.2)
    plt.legend(loc=0)
    plt.show()


if __name__ == '__main__':
    main()