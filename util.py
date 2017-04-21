def one_hot_encoder(n, vec_size):
	import numpy as np 

	ret = np.zeros(vec_size)
	ret[n] = 1
	return ret


def plot(rewards, q):
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	plt.plot(rewards, 'b', q, 'r')
	plt.savefig('models/index.jpg')