import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.style.use('ggplot')


def create_data():
	means = np.random.uniform(low=-3.0, high=3.0, size=(2,2))
	stddevs = np.random.uniform(low=0.3, high=0.6, size=(2,2))
	data = []
	for mean, stddev in zip(means, stddevs):
		data.extend(stddev**2 * np.random.randn(15, 2) + mean)
	return np.asarray(data), means, stddevs

	
def gaus(x, m, s):
	return np.exp((-(m-x)**2)/2*s**2) / ((2 * np.pi) ** 0.5 * s)
	

class GMM():

	cnt_plot = 0
	
	def __init__(self, k, dim):
		self.means = []
		self.stddevs = []
		self.c_k = np.asarray([1/k for _ in range(k)])
		self.dim = dim
		num = np.abs(np.random.randn(1))
		for _ in range(k):
			self.means.append(np.random.randn(dim))
			self.stddevs.append(np.repeat(num, dim))
		self.fig = plt.figure()
	
	def prepplot(self, data, text='time axis', means=None, stddevs=None):
		if self.dim != 2:
			print('Only setup to plot 2D data, returning.')
			return
		self.cnt_plot += 1
			
		ax = self.fig.add_subplot(2,3,self.cnt_plot)
		ax.set_xlim([-5, 5])
		ax.set_ylim([-5, 5])
		ax.set_xlabel(text)
		ax.scatter(data[:,0], data[:,1], s=25)	
		if means is None:
			means, stddevs = self.means, self.stddevs			
		for mean, stddev in zip(means, stddevs):
			ellipse = Ellipse(xy=mean, width=stddev[0], height=stddev[1], fill=False, color='b')
			ax.add_artist(ellipse)
	
	def plot(self):
		plt.tight_layout()
		plt.show()
	
	def train(self, data):
		resp = []  # responsibilities
		for sample in data:
			r = []
			for i, (mean, stddev) in enumerate(zip(self.means, self.stddevs)):
				sample_r = self.c_k[i]
				for j in range(self.dim):
					sample_r *= gaus(sample[j], mean[j], stddev[j])
				r.append(sample_r)
			r = np.asarray(r) 
			r /= np.sum(r)
			resp.append(r)	
		resp = np.asarray(resp)  # Shape (sample, cluster)
		
		R = np.sum(resp, axis=0)
		tmp = resp.T@data
		self.means = ((tmp).T / R).T  # Transposes for division across row.

		sum_R = np.sum(R)
		for i, mean in enumerate(self.means):
			diff = np.square(data - mean)
			self.stddevs[i] = np.sqrt(resp[:, i].dot(diff) / R[i])
			self.c_k[i] = R[i]/sum_R
		
		
def main():
	data, means, stddevs = create_data()
	gmm = GMM(2, 2)
	gmm.prepplot(data, text='t=0')
	gmm.train(data)
	gmm.prepplot(data, text='t=1')
	for i in range(61):
		gmm.train(data)
		if i % 20 == 0:	
			gmm.prepplot(data, text='t={}'.format(i+2))
	
	gmm.plot()
	
	
main()
		