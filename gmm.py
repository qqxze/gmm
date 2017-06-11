import sys
import numpy as np
from numpy.random import permutation
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import colors as mcolors
plt.style.use('ggplot')

np.random.seed(42)
np.seterr(all='raise')


def create_data():
    means = np.random.uniform(low=-4.0, high=4.0, size=(2,2))
    stddevs = np.random.uniform(low=1.0, high=1.4, size=(2,2))
    data = []
    for mean, stddev in zip(means, stddevs):
        data.extend(stddev**2 * np.random.randn(20, 2) + mean)
    return permutation(np.asarray(data)), means, stddevs


def logg(x, m, s):
    return -(m-x)**2 / (2*s**2) + np.log(1/((2 * np.pi) ** 0.5 * s))


class MultiPlot():

    def __init__(self, shape):
        self.fig = plt.figure(figsize=(12,8,))    
        self.cnt_plot = 0
        self.shape = shape
        self.colors = list(mcolors.BASE_COLORS.keys())
        self.colors.remove('r')
        self.len_colors = len(self.colors)

    def add_plot(self, data, text='time axis', means=None, stddevs=None):
        self.cnt_plot += 1

        ax = self.fig.add_subplot(*self.shape, self.cnt_plot, aspect='equal')
        ax.set_xlim([-7, 7])
        ax.set_ylim([-7, 7])
        ax.set_xlabel(text)

        ax.scatter(data[:,0], data[:,1], s=15, marker='x')    
        if means is not None:
            for i, (mean, stddev) in enumerate(zip(means, stddevs)):
                s1, s2 = mean[0], mean[1]
                color = self.colors[i % self.len_colors]
                ellipse = Ellipse(xy=(s1, s2,), width=stddev[0], height=stddev[1], color=color, fill=False, label='{s1:=1.2f},{s2:=1.2f}'.format(s1=s1, s2=s2))
                ax.add_patch(ellipse)
        ax.legend()


class GMM():
   
    def __init__(self, k, dim, data):
        self.means = []
        self.stddevs = []
        self.c_k = np.asarray([1/k for _ in range(k)])
        self.dim = dim

        len_data = len(data)
        len_subset = len_data // k
        
        means_init, stddevs_init = [], []
        for i in range(k):
            data_subset = data[i * len_subset: (i+1) * len_subset]
            means_init.append(np.mean(data_subset, axis=0))
            stddevs_init.append(np.std(data_subset, axis=0))
        for iter in range(3):
            data_clusters = [[] for _ in range(k)]
            for sample in data:
                diffs = []
                for mean in means_init:
                    diffs.append(np.sum((sample - mean) ** 2) ** 0.5)
                data_clusters[np.argmax(diffs)].append(sample)
            for i in range(k):
                data_cluster = data_clusters[i]
                if data_cluster:
                    means_init[i] = np.mean(data_cluster, axis=0)
                    if iter == 2:
                        stddevs_init[i] = np.std(data_cluster, axis=0)

        self.means = np.asarray(means_init)
        self.stddevs = np.asarray(stddevs_init)

    def trainlog(self, data):
        resp = np.empty((len(data), len(self.c_k)))  # responsibilities. Shape (sample, cluster)
        for n, sample in enumerate(data):
            for i, (mean, stddev) in enumerate(zip(self.means, self.stddevs)):
                sample_r = np.log(self.c_k[i])
                for j in range(self.dim):
                    sample_r += logg(sample[j], mean[j], stddev[j])
                if sample_r < -60.0:
                    sample_r = -60.0
                elif sample_r >= -1e-5:
                    sample_r = -1e-5

                resp[n, i] = sample_r
            sum_r = np.log(np.sum(np.exp(resp[n])))  # resp[n] contains log(a); log(b), we need log(a + b)
            resp[n] -= sum_r
        resp = np.exp(resp)
        R = np.sum(resp, axis=0)
        self.means = ((resp.T@data).T / R).T  # Transposes for division along row (cluster) (default is column).

        sum_R = np.sum(R)

        for i, mean in enumerate(self.means):
            diff = data - mean
            if np.sum(self.stddevs[i]) > 0.01:
                self.stddevs[i] = np.sqrt(resp[:, i].dot(np.square(diff)) / R[i])
            self.c_k[i] = R[i]/sum_R
            
def main():
    iters = 7

    data, _, _ = create_data()
    gmm = GMM(2, 2, data=data)

    m, s, c_ks = np.copy(gmm.means), np.copy(gmm.stddevs), np.copy(gmm.c_k)
    mul_plot = MultiPlot((2, 4,))
    mul_plot.add_plot(data, text='t=0', means=m, stddevs=s)
    for i in range(iters):
        gmm.trainlog(data)
        nm, ns = np.copy(gmm.means), np.copy(gmm.stddevs)
        mul_plot.add_plot(data, text='t={}'.format(i+1), means=nm, stddevs=ns)

    plt.tight_layout()
    plt.show()

    
main()