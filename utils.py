from scipy.special import psi # digamma function
import numpy as np

def dirichlet_expectation(parameters):
	'''
	Computes the expectation of the log of variational parameters.
	'''
	# if a vector is passed instead of a matrix
	if len(parameters.shape) == 1:
		E_log_param = psi(parameters) - psi(np.sum(parameters))

	# if a matrix is passed, we subtract the row sum from each row element
	E_log_param = psi(parameters) - psi(np.sum(parameters, axis=1)).reshape(-1, 1)

	return E_log_param


def get_top_words(beta, vocab, n_topic = None, n_top_words = 10):
	if n_topic is None:
		for i, topic_dist in enumerate(beta):
			topic_words = np.argsort(topic_dist)[:-(n_top_words+1):-1]
			top_words = []
			for j in topic_words:
				top_words.append(vocab[j])
			print('Topic {}: {}'.format(i, ' '.join(top_words)))

	if type(n_topic) == int:
		try:
			topic_dist = beta[n_topic, :]
			topic_words = np.argsort(topic_dist)[:-(n_top_words+1):-1]
			top_words = []
			for j in topic_words:
				top_words.append(vocab[j])
			print('Topic {}: {}'.format(i, ' '.join(top_words)))
		except Exception:
			raise ValueError('The argument must be the index of the topic.')


def calc_term_scores(beta):
	K = beta.shape[0]
	return beta * np.log(beta / np.power(np.sum(beta, axis=0), 1/K))

def calc_var_auc(AUC, n0, n1):
    
    Q1 = AUC/(2-AUC)
    
    Q2 = 2*(AUC**2) / (1+AUC)
    
    denominator = 1/(n0*n1)
    
    numerator = AUC*(1-AUC) + (n1 - 1)*(Q1 - AUC**2) + (n0 - 1)*(Q2 - AUC**2)
    
    numerator = numerator**0.5
    
    variance = numerator * denominator
    
    return variance





