import sys
import numpy as np
from numpy.random import shuffle
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
plt.style.use('ggplot')

np.random.seed(15)
np.seterr(all='raise')


def create_data():
    means = np.random.uniform(low=-5.0, high=5.0, size=(2,2))
    stddevs = np.random.uniform(low=0.4, high=0.6, size=(2,2))
    data = []
    for mean, stddev in zip(means, stddevs):
        data.extend(stddev**2 * np.random.randn(30, 2) + mean)
    return np.asarray(data), means, stddevs


def logg(x, m, s):
    return -(m-x)**2 / (2*s**2) + np.log(1/((2 * np.pi) ** 0.5 * s))


class MultiPlot():

    def __init__(self, shape):
        self.fig = plt.figure(figsize=(12,8,))    
        self.cnt_plot = 0
        self.shape = shape

    def add_plot(self, data, text='time axis', means=None, stddevs=None):
        self.cnt_plot += 1

        ax = self.fig.add_subplot(*self.shape, self.cnt_plot)
        ax.set_xlim([-7, 7])
        ax.set_ylim([-7, 7])
        ax.set_xlabel(text)

        ax.scatter(data[:,0], data[:,1], s=20, marker='x')    
        if means is not None:
            for mean, stddev in zip(means, stddevs):
                s1, s2 = mean[0], mean[1]
                ellipse = Ellipse(xy=(s1, s2,), width=stddev[0], height=stddev[1], color='b', fill=False, label='{s1:=1.2f},{s2:=1.2f}'.format(s1=s1, s2=s2))
                ax.add_patch(ellipse)
        ax.legend()


class GMM():
   
    def __init__(self, k, dim, data):
        self.means = []
        self.stddevs = []
        self.c_k = np.asarray([1/k for _ in range(k)])
        self.dim = dim
        num = np.abs(np.random.randn(1))
        shuffle(data)
        len_subset = len(data) // k
        for i in range(k):
            data_subset = data[i * len_subset: (i+1) * len_subset]
            self.means.append(np.mean(data_subset, axis=0))
            self.stddevs.append(np.diag(np.cov(data_subset, rowvar=False)))
        self.means = np.asarray(self.means)
        self.stddevs = np.asarray(self.stddevs)

    def trainlog(self, data):
        resp = np.empty((len(data), len(self.c_k)))  # responsibilities. Shape (sample, cluster)
        for n, sample in enumerate(data):
            for i, (mean, stddev) in enumerate(zip(self.means, self.stddevs)):
                sample_r = np.log(self.c_k[i])
                for j in range(self.dim):
                    sample_r += logg(sample[j], mean[j], stddev[j])
                sample_r = -80.0 if sample_r < -80.0 else sample_r
                resp[n][i] = sample_r
            sum_r = np.sum(resp[n])
            resp[n] -= sum_r
        resp = np.exp(resp)
        R = np.sum(resp, axis=0)
        # Transposes for division across row (default is column).
        self.means = ((resp.T@data).T / R).T  

        sum_R = np.sum(R)
        for i, mean in enumerate(self.means):
            diff = data - mean
            if np.sum(self.stddevs) > 0.01:
                self.stddevs[i] = np.sqrt(resp[:, i].dot(np.square(diff)) / R[i])
            self.c_k[i] = R[i]/sum_R
            
def main():
    iters = 7

    data, means, stddevs = create_data()
    gmm = GMM(2, 2, data=data)
    #gmm.means, gmm.stddevs = means, stddevs

    m, s, c_ks = np.copy(gmm.means), np.copy(gmm.stddevs), np.copy(gmm.c_k)
    mul_plot = MultiPlot((2, 4,))
    mul_plot.add_plot(data, text='t=0', means=m, stddevs=s)
    for i in range(5*iters):
        gmm.trainlog(data)
        nm, ns = np.copy(gmm.means), np.copy(gmm.stddevs)
        if i % 5 == 0:
            mul_plot.add_plot(data, text='t={}'.format(i+1), means=nm, stddevs=ns)

    gmm2 = GaussianMixture(2, covariance_type='diag', reg_covar=1e-6, weights_init=c_ks, means_init=m, precisions_init=1.0/s)
    precs = gmm2.precisions_init
    means, stddevs = np.copy(gmm2.means_init), 1.0/precs

    mul_plot2 = MultiPlot((1, 2,))
    mul_plot2.add_plot(data, text='t=0', means=means, stddevs=stddevs)
    
    gmm2.fit(data)

    means, stddevs = np.copy(gmm2.means_), np.copy(gmm2.covariances_)
    mul_plot2.add_plot(data, text='t=1', means=means, stddevs=stddevs)    

    plt.tight_layout()
    plt.show()

    
main()





