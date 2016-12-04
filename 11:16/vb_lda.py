'''
(с) Dat Nguyen (dat.nguyen@cantab.net)
'''

import numpy as np
from scipy.special import gammaln, psi
from scipy.misc import logsumexp
import time

import parser
from utils import *

class lda(object):
	'''
	Mean-field variational inference algorithm for Latent Dirichlet Allocation.

	Specifically, the implementation follows the notation in Blei and Lafferty (2009).

	---
	References:

	Blei, Ng and Jordan (2003), Latent Dirichlet Allocation;
	Blei and Lafferty (2009), Topic Models.
	'''

# 1. Initialization
	### Decide how to pass the doc_array
	def __init__(self, K, alpha, eta):
		# assert len(alpha) == K or len(list(alpha)) == 1;
		self.alpha = alpha # if len(alpha)==K else alpha*np.ones(self.num_topics);
		self.num_topics = K;
		# prior on Beta
		self.eta = eta;

	def _initialize(self, doc_term_ids, doc_term_counts, vocab):
		'''
		document parameters
		'''

		#self.alpha = np.array([1/K]*K)
		#self._phi = []
		self.vocab_size = len(vocab) 

		# assert len(doc_term_ids) == len(doc_term_counts)
		self.num_docs = len(doc_term_ids)
		
		# assert len(self.alpha) == self.num_topics
		self.alpha = self.alpha + np.zeros(self.num_topics)
		# assert len(self.eta) == self.vocab_size
		self.eta = self.eta + np.zeros(self.vocab_size)

		# self._gamma = 1/self.num_topics * np.ones((self.num_docs, self.num_topics))
		self._gamma = np.random.gamma(100., 1. / 100., 
		(self.num_docs, self.num_topics));

		self._lambda = np.random.gamma(100., 1. / 100., 
		(self.num_topics, self.vocab_size));
		# TBC: 'seed' docs to initialize (p.11)
		# self._lambda = 1/self.vocab_size * np.ones((self.num_topics, self.vocab_size))

# 2. Learning

	def fit(self, corpus, 
		max_iter = 10, 
		convergence_thres = 1e-3,
		gamma_max_iter = 10,
		convergence_criterion = 1e-3,
		**kwargs):

		docs, vocab = parser._parse_vocab(corpus, **kwargs)
		print('Created the vocabulary. # terms:', len(vocab.keys()))

		# store the vocabulary as a dictionary to access word strings
		self._vocab = vocab

		doc_term_ids, doc_term_counts = parser._parse_corpus(docs, vocab)
		print('Successfully parsed the corpus. # docs: ', len(doc_term_ids))

		self._initialize(doc_term_ids, doc_term_counts, vocab)

		# Mean-field algorithm (variational EM)
		counter = 1
		elbo = 0
		previous_elbo = elbo + convergence_thres + 1
		clock = time.time()

		while abs(elbo - previous_elbo) > convergence_thres and counter < max_iter+1:

			previous_elbo = elbo

			phi_over_all_d, elbo_without_beta = self.e_step(doc_term_ids, 
										doc_term_counts,
										gamma_max_iter,
										convergence_criterion);

			elbo = self.m_step(phi_over_all_d, elbo_without_beta);

			if counter == 1 or counter % 5 == 0:
				print('Successfully completed %d iteration(s) in %ds.' %(counter, time.time() - clock))
				print('ELBO =', elbo)
			counter += 1
		print('Learning completed!\nTotal time taken: %ds\nELBO = %f' %(time.time() - clock, elbo))

		theta = self._gamma / np.sum(self._gamma, axis=1).reshape(-1, 1)

		beta = self._lambda / np.sum(self._lambda, axis=1).reshape(-1, 1)

		return theta, beta #, phi_over_all_d

# 2.1 E-Step: coordinate ascent in gamma and phi

	def e_step(self, doc_term_ids, doc_term_counts, 
				gamma_max_iter = 10,
				convergence_criterion=1e-3):
		'''
		iterate through each document and update variational parameters
		gamma and phi until convergence.

		this method also returns those components of the ELBO
		that require summation over documents,
		which is convenient to do inside the loop as we update
		the variational parameters due to the assumption of
		independence between documents.

		---
		Reference:

		Blei, Ng and Jordan (2003), Appendix A.3;
		Blei and Lafferty (2009), p.9-11.
		'''

		# Elog[p(theta | alpha)] + Elog[p(Z | theta)]
			# - Elog[q(theta)] - Elog[q(z)]
		elbo_without_beta = 0

		# keep phid KxV because we will later use it 
		# to update lambda which is also KxV
		phi_over_all_d = np.zeros((self.num_topics, self.vocab_size)) # KxV
		# phi_over_all_d = np.zeros((self.vocab_size, self.num_topics)) # VxK

		# expectation of log beta under the variational Dirichlet (p. 10)
		# E_log_beta = psi(lambda) - psi(np.sum(lambda, axis=1))
		E_log_beta = psi(self._lambda) - psi(np.sum(self._lambda, axis = 1)).reshape(-1, 1) # KxV matrix
		
		for d in range(self.num_docs):

			# Term ids that appear in document d
			term_ids = np.array(doc_term_ids[d], dtype=np.int)

			# Term counts (for each unique term) in document d
			term_counts = np.array(doc_term_counts[d], dtype=np.int)

			# Number of unique terms in document d
			total_term_count = len(term_ids)

			# Total number of words in document d
			total_word_count = np.sum(term_counts, dtype=np.int)

			# self._gamma[d, :] = self.alpha + total_word_count/self.num_topics

			gammad = self._gamma[d, :]

			for iterr in range(gamma_max_iter):

				lastgamma = gammad;

				# Calculate phi first
				# Log transformation to avoid the exponential function (slow)
				# since gamma[d,k] is the same for all v, we repeat it using np.tile

				# T_d x K matrix, where T_d = total_term_count
				# This is because updates for phi are repeated for the same term (p. 11)
				log_phid = np.tile(psi(self._gamma[d, :]), reps=(total_term_count, 1)) + E_log_beta[:, term_ids].T;


				# the equation above is only true proportionally (p. 9)
				# since we took logs, we will substract the normalizer (not divide)
				log_phid_normalizer = logsumexp(log_phid, axis=1) # sum over topics (cols)
																# since for each topic, sum(probs) = 1

				log_phid = log_phid - log_phid_normalizer.reshape(-1, 1);

				# update for gamma
				# gammad = self.alpha + np.dot(term_counts.reshape(1, -1), np.exp(log_phid))
				gammad = self.alpha + np.array(np.sum(np.exp(log_phid + np.log( term_counts.reshape(-1, 1) )), axis = 0)); # sum over rows to get a 1xK vector

				assert len(gammad) == self.num_topics

				if np.mean(abs(gammad - lastgamma)) < convergence_criterion:
					break;

			self._gamma[d, :] = gammad

			phi_over_all_d[:, term_ids] += np.exp(log_phid + np.log( term_counts.reshape(-1, 1) ) ).T;
			# phi_over_all_d[term_ids, :] += np.exp(log_phid + np.log( term_counts.reshape(-1, 1) ) )

			# NOTE: terms involving psi(gamma) get cancelled
			# because gammad = alpha + \sum_n(phi_dn)
			elbo_without_beta += gammaln(np.sum(self.alpha)) - np.sum(gammaln(self.alpha))

			elbo_without_beta += np.sum(gammaln(gammad)) - gammaln(np.sum(gammad));

			# dot product gives summation over N_d
			# afterwards we sum over K to get Elog[q(z)]
			elbo_without_beta -= np.sum(np.dot(term_counts, (np.exp(log_phid) * log_phid)))

		return phi_over_all_d, elbo_without_beta

# 2.2 M-step: coordinate ascent in lambda

	def m_step(self, phi_over_all_d, elbo_without_beta):
		'''
		update the variational parameter lambda.
		'''

		self._lambda = self.eta + phi_over_all_d;
		# self._lambda = self.eta + phi_over_all_d.T

		elbo = elbo_without_beta

		# (-)Elog[q(lambda)]
		elbo += np.sum( np.sum(gammaln(self._lambda), axis=1) - gammaln(np.sum(self._lambda, axis=1)) )

		# Note: terms containing psi(eta) get cancelled out
		# Elog[p(w | Z, beta)] also gets cancelled out

		# Elog[p(beta | eta)] is analogous to Elog[p(theta | alpha)]
		# except we need to sum over K, rather than D
		# since Elog[p(beta | eta)] does not depend on K
		# we can simply multiply by K
		elbo += self.num_topics * ( gammaln(np.sum(self.eta)) - np.sum(gammaln(self.eta)) );

		return elbo



























		





			



