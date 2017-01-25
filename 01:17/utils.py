from scipy.special import psi # digamma function
import numpy as np

def dirichlet_expectation(variational_params):
	'''
	Computes the expectation of the log of variational parameters.
	'''
	# if a vector is passed instead of a matrix
	if len(variational_params.shape) == 1:
		E_log_param = psi(variational_params) - psi(np.sum(variational_params))

	# if a matrix is passed, we subtract the row sum from each row element
	E_log_param = psi(variational_params) - psi(np.sum(variational_params, axis=1)).reshape(-1, 1)

	return E_log_param


def get_top_words(topic_distr, vocab, n_topic = None, n_top_words = 10):
	if n_topic is None:
		for i, topic_dist in enumerate(topic_distr):
			topic_words = np.argsort(topic_dist)[:-(n_top_words+1):-1]
			top_words = []
			for j in topic_words:
				top_words.append(vocab[j])
			print('Topic {}: {}'.format(i, ' '.join(top_words)))

	if type(n_topic) == int:
		try:
			topic_dist = topic_distr[n_topic, :]
			topic_words = np.argsort(topic_dist)[:-(n_top_words+1):-1]
			top_words = []
			for j in topic_words:
				top_words.append(vocab[j])
			print('Topic {}: {}'.format(i, ' '.join(top_words)))
		except Exception:
			raise ValueError('The first argument must be the index of the topic and the second -- the number of top words to be shown.')




