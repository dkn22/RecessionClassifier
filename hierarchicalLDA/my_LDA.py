"""
The Gibbs sampling algorithm that learns Latent Dirichlet Allocation.

DGM for a document: each document has N_d words, each word is generated independently;
    				for each document d and word position n, z_dn is sampled from THETA_d, the topic distribution;
    				for a given sampled topic z_dn, the word w_dn is sampled from BETA_(z_dn).
"""

import numpy as np
from numpy.random import choice
from numpy.random import multinomial
from numpy.random import dirichlet
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 

### 3. Hierarchical LDA ###

'''
Hierarchical prior for the LDA, finite depth
'''

class hLDA(object):
	def __init__(self, tree_depth, alpha, gamma, eta):
		'''
		gamma: the parameter for the (base) Dirichlet from which the prior vector ('m') is drawn.
				in the notes' notation, this is alpha'.
		tree_depth: a parameter for the number of levels in a hierarchy.
		'''
		self.gamma = gamma
		self.alpha = alpha
		self.k = tree_depth
		self.eta = eta

	def prior(self, corpus, seed=None):
		self.num_docs = len(corpus)
		dtm = DocTermMatrix(corpus).dataframe()
		# Document-specific vectors that record the multiplicity of each label
		local_count = np.zeros((self.num_docs, self.k))
		# Global vector which keeps tracks of all draws from the base measure
		global_count = np.zeros(self.k)

		if seed is not None:
			np.random.seed(seed)

		# Simulating from the hierarchical prior for the first document
		sample = {}
		sample[0] = multinomial(1,[1/self.k]*self.k).nonzero()[0][0] # sample s1
		R = set(sample.values())
		M = 1	
		z = {}
		# set the topic of the first word in the first doc to s1
		z[(0,0)] = sample[0]
		global_count[0, sample[0]] += 1
		local_сount[0, sample[0]] += 1
		for n, w in enumerate(DocTermMatrix.word_index(dtm[0,:])):
			draw = hLDA.roll_die_first_doc()
			if draw is not None and draw <= self.k:
				z[(0,n+1)] = draw
				local_count[0, draw] += 1
			elif draw > self.k and draw <= 2*self.k:
				sample[M+1] = draw - self.k # assign the s_{Mn+1} = s

				# update local and global count?
				local_count[0, sample[M+1]] += 1
				global_count[0, sample[M+1]] += 1

				M += 1
				# set z_{n+1} equal to?
				z[(0, n+1)] = sample[M+1]
			elif draw > 2*self.k:
				new_s = multinomial(1,[1/self.k]*self.k).nonzero()[0][0]
				sample[M+1] = new_s
				M += 1
				global_count[0, new_s] += 1
				local_count[0, new_s] += 1
				# set z_{n+1} equal to?
				z[(0, n+1)] = new_s
			R = set(sample)

		# hierarchical prior across documnents
		for idx, bow in dtm[1:].iterrows(): # bow = bag-of-words
			for n, w in enumerate(DocTermMatrix.word_index(bow)):
				if n == 0:
					draw = hLDA.roll_die_first_word()
					if draw <= self.k:
						z[(idx,n)] = draw # double-check
						local_count[idx, draw] = 1
					else:
						sample[M+1] = multinomial(1,[1/self.k]*self.k).nonzero()[0][0]
						z[(idx,n)] = sample[M+1] # is this the s in the notes?
						M += 1
				else:
					draw = hLDA.roll_die_nth_word()
					if draw <= self.k:
						# need to align n's
						z[(idx, n+1)] = draw
						global_count[idx, draw] += 1
						local_count[idx, draw] += 1
					else:
						sample[M+1] = multinomial(1,[1/self.k]*self.k).nonzero()[0][0]
						z[(idx,n+1)] = sample[M+1]
						global_count[idx, sample[M+1]] += 1
						local_count[idx, sample[M+1]] += 1
						M += 1



	@staticmethod
	def roll_die_first_doc(n_d=local_count, n_g=global_count, n=n, M=M):
		# probabilities of picking one of s
		probs_old_draws = local_count[0, :] / (self.alpha + n)

		# probabilities of a new draw being set to s
		probs_new_draw_s = global_count[0,:] / (self.gamma + n)
		probs_new_draw_s *= (self.alpha / (self.alpha + n))

		# probability of a new draw from the base measure
		prob_new_from_base = (self.alpha / (self.alpha + M)) * (self.gamma / (self.gamma + M))
		
		probabilities = list(probs_old_draws) + list(probs_new_draw_s) + list(prob_new_from_base)
		draw = choice(len(probabilities), p=probabilities)

		return draw

	@staticmethod
	def roll_die_first_word(n_d=local_count, n_g=global_count, M=M, doc_idx = idx):
		# probabilities of picking one of s
		probs_old_draws = global_count[idx, :] / (self.gamma + M)

		# probability of a new draw
		prob_new_draw = self.gamma / (M + self.gamma)

		probabilities = list(probs_old_draws) + list(prob_new_draw)
		draw = choice(len(probabilities), p=probabilities)
		return draw

	@staticmethod
	def roll_die_nth_word(n_d=local_count, n_g=global_count, n=n, M=M, doc_idx=idx):
		# probabilities of picking one of s
		probs_old_draws = local_count[idx, :] / (self.alpha + n)

		# probability of a new draw
		prob_new_draw = self.alpha / (self.alpha + n)

		probabilities = list(probs_old_draws) + list(prob_new_draw)
		draw = choice(len(probabilities), p=probabilities)
		return draw


### 1. Defining the class and methods for the Document-Term matrix ###

class DocTermMatrix(object):
	def __init__(self, corpus, **kwargs):
		'''
		corpus: a collection of raw documents
		with **kwargs you can pass arguments of CountVectorizer
		creates a doc-term matrix instance (as scipy sparse matrix)
		'''
		
		vectorizer = CountVectorizer(**kwargs)
		self.num_docs = len(corpus)
		self.vocab_size = len(vectorizer.vocabulary_)
		self.shape = (self.num_docs, self.vocab_size)
    	self.matrix = vectorizer.fit_transform(docs)
	
	def dataframe(self):
		'''
		returns the doc-term matrix as pandas dataframe
		'''
		return pd.DataFrame(x1.toarray(), columns = vectorizer.get_feature_names())

	def length(self):
		"""
		Calculates the number of words in each document, i.e. its length.
		Returns an array of the same size as the corpus.
		"""
		# lengths_list = np.zeros(self.row_dim)
		lengths_list = self.df().apply(lambda x: len(x.nonzero()[0]), axis=1)
		return lengths_list

	@classmethod
	def word_index(cls, doc_row):											# in a document-term matrix, the order of the words is lost: is that a problem? (refer to DGM)
		"""
		doc_row: document vector (of size V, vocab-size)

		Returns an array of repeated indices/terms for the words in the document;
		the indices/terms are repeated the same number of times as the number of occurrences for the corresponding word;
		the length of the array is thus equal to the document length.
		"""
		# doc_row = np.array(doc_row)
		for idx in doc_row.nonzero()[0]:
        	for i in range(int(doc_row[idx])):
            	yield idx

### 2. The LDA class including both uncollapsed and collapsed Gibbs sampler ###

class LDA(object):
	def __init__(self, num_topics, alpha, eta):
		"""
		num_topics:	number of topics
		alpha:		Dirichlet parameter for the topic distribution 
        			THETA with T categories (for each document d)
    	eta:    		Dirichlet parameter for the word distribution 
               		BETA with W categories (for each topic t)
		"""
		self.num_topics = num_topics
		self.alpha = alpha
		self.eta = eta

	def _initialize(self, corpus):
		"""
		doc_matrix: a DxV document-term matrix

		Initialises the count variables N_dk, N_kv, N_k, N_d.
			K: topic
			D: document
			V: term
		"""
		
		doc_matrix = DocTermMatrix(corpus)
		num_docs, vocab_size = doc_matrix.shape

		# number of terms/words in document D that have topic allocation K
		self.ndk = np.zeros((num_docs, self.num_topics))
		# number of times topic K allocaition variables generate term/word V
		self.nkv = np.zeros((self.num_topics, vocab_size))
		# number of terms/words generated by topic K
		self.nk = np.zeros(self.num_topics)
		# number of terms/words in document D
		self.nd = doc_matrix.length() # array of size num_docs

		# The Dirichlet prior for the document-topic distribution
		theta = dirichlet(self.alpha * np.ones(self.num_topics), size = num_docs)

		# The Dirichlet prior for the topic-term distribution
		beta = dirichlet(self.eta * np.ones(vocab_size), size = self.num_topics)

		# Initialize the topic assignment variables as a dictionary
			# key corresponds to (d,n): nth word in document d
			# value = {0,...,K-1} is the topic assigned to that word
		z = {} 															# Documents have different lengths, so a matrix would be an inappropriate data structure?
																		# check correct indices are past, especially w vs. i
		for d in range(num_docs):
			# i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
			for i, w in enumerate(DocTermMatrix.word_index(doc_matrix[d,:])):
				# total iterations of i will equal the document length
				# the set of w includes all terms in document d
				z[(d,i)] = multinomial(1, theta[d]).nonzero()[0][0]
				self.ndk[d,z[(d,i)]] += 1
				self.nkv[z[(d,i)],w] += 1
				self.nk[z[(d,i)]] += 1

	def run_Gibbs(self, doc_matrix, maxIter):
		"""
		Uncollapsed Gibbs sampling.

		matrix: the document-term matrix.
		maxIter: the number of iterations to run.

		One could construct predictive distributions for theta and beta from the posterior samples (like we do in the collapsed Gibbs).
		"""
		num_docs, vocab_size = doc_matrix.shape
		self._initialize(doc_matrix)
		theta_posterior = theta 	# DxK matrix
		beta_posterior = beta 		# KxV matrix
		theta_samples = []
		beta_samples = []

																		# The number of loops seems very inefficient
		for iterr in range(maxIter):
			for k in range(self.num_topics):
				# Posterior for BETA
				beta_posterior[k,:] = dirichlet(self.eta + self.nkv[k,:])
			for d in range(num_docs):
				# Posterior for THETA
				theta_posterior[d,:] = dirichlet(self.alpha + self.ndk[d,:])
			for d in range(num_docs):
				for i, w in enumerate(DocTermMatrix.word_index(doc_matrix[d,:])):
					self.update_counts(True, d, i, w)
					z[(d,i)] = self.sample_topic_assignment(d, w, theta_posterior, beta_posterior)
					# update counts
					self.update_counts(False, d, i, w)
			# Burn-in and thinning interval set to 10 iterations
			if (iterr + 1)%10 == 0:
				theta_samples.append(theta_posterior)
				beta_samples.append(beta_posterior)
			if iterr == maxIter:
				theta_samples.append(theta_posterior)
				beta_samples.append(beta_posterior)
																		
		return z, theta_samples, beta_samples, self.ndk, self.nkv, self.nk 


	def sample_topic_assignment(self, d, w, theta, beta):
		zdi_probs = np.zeros(self.num_topics)
		for k in range(self.num_topics):
			zdi_probs[k] = theta[d,k]*beta[k,w] / np.dot(theta[d,:],beta[:,w])
		# Why do we not assign the topic with the highest probability?
		zdi = multinomial(1, zdi_probs).nonzero()[0][0]
		return zdi


	def run_collapsedGibbs(self, doc_matrix, maxIter):
		"""
		Collapsed Gibbs sampling.

		Tends to be more efficient because we reduce the dimensionality of the problem
		by integrating out THETA and BETA.
		"""
		num_docs, vocab_size = doc_matrix.shape
		self._initialize(doc_matrix)
		theta_samples = []
		beta_samples = []

		for iterr in range(maxIter):
			for d in range(num_docs):
				for i, w in enumerate(DocTermMatrix.word_index(doc_matrix[d,:])):
					self.update_counts(True, d, i, w)
					z[(d,i)] = self.sample_topic_assignment_collapsed(d,w)
					self.update_counts(False, d, i, w)

			# Burn-in and thinning interval set to 10 iterations
			if (iterr + 1)%10 == 0:
				theta_samples.append(self.compute_theta())
				beta_samples.append(self.compute_beta())
			# Final update
			if iterr == maxIter:
				theta_samples.append(self.compute_theta())
				beta_samples.append(self.compute_beta())
		return z, theta_samples, beta_samples, self.ndk, self.nkv, self.nk

	def update_counts(self, decrease_count, d, i, w):
		"""
		decrease_count: Boolean
		d: 		 		document number
		i:		 		ith word [unordered] in document d
		w:		 		term w in document d

		Updates count variables to run collapsed sampling equation for LDA.
		"""
		if decrease_count:
			self.ndk[d,z[(d,i)]] -= 1
			self.nkv[z[(d,i)],w] -= 1
			self.nk[z[(d,i)]] -= 1
		else:
			self.ndk[d,z[(d,i)]] += 1
			self.nkv[z[(d,i)],w] += 1
			self.nk[z[(d,i)]] += 1

	def sample_topic_assignment_collapsed(self, d, w):
		# THETA and BETA have been integrated out and so do not enter as parameters.
		
		zdi_probs = np.zeros(self.num_topics)
		for k in range(self.num_topics):
			# self.nkv.shape[1] = column dimension of self.nkv = vocab_size
			zdi_probs[k] = (self.ndk[d,k] + self.alpha) * (self.nkv[k,w] + self.eta) / (self.nkv[k, :].sum() + self.eta * self.nkv.shape[1])
		zdi = multinomial(1, zdi_probs).nonzero()[0][0]
		return zdi

	def compute_theta(self):
		"""
		Approximate the predictive document-topic distribution THETA; vectorized implementation.
		"""
		numerator = self.ndk + self.alpha 						# DxK matrix
		denominator = np.sum(numerator, axis=1)[:, np.newaxis]	# Dx1 vector: total sum of elements for each row of numerator

		# Divide each element in row d of numerator by element d of denominator
		theta = numerator / denominator
		return theta

	def compute_beta(self):
		"""
		Approximate the predictive topic-term distributions BETA; vectorized implementation.
		"""
		numerator = self.nkv + self.eta 							# KxV matrix
		denominator = np.sum(numerator, axis=1)[:, np.newaxis]	# Kx1 vector: total sum of elements for each row of numerator

		beta = numerator / denominator
		return beta

