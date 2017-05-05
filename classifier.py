from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from topicmodel import LDA
from operator import itemgetter
from preprocess import Parser
from scipy.misc import logsumexp


def train_test_sample(full_corpus, merged_corpus=None,
					n_samples=100,
					primary_key='meeting',
					merge_train_docs = False):
	'''
	Splits the text data into train set and test set.
	Test set will consist of one (merged) document.

	Train documents are not merged unless specified.
	'''
	# grouped_data = full_data.groupby('meeting', as_index=False)
	# full_minutes = pd.concat([grouped_data['text'].apply(' '.join), grouped_data.first()], axis=1)
	# full_minutes.drop(['text', 'seq'], axis=1, inplace=True)
	# full_minutes.rename(columns={0: 'text'}, inplace=True)

	permuted = np.random.permutation(full_corpus[primary_key])

	for i in range(n_samples):
		date = permuted[i]

		test_data = merged_corpus[merged_corpus[primary_key] == date]

		if type(test_data) == str:
			test_data = [out_of_sample_data]

		train_data = full_corpus[full_corpus[primary_key] != date]

		if merge_train_docs:
			train_data = merged_corpus[merged_corpus[primary_key] != date]

		yield (train_data, test_data)



class DiscriminativeClassifier(object):
	'''
	A logistic classifier that takes
	LDA doc-topic vectors as features.
	'''

	def __init__(self):
		print('A discriminative classifier is initialised. By default, it uses ridge regression.')

	def fit(self, X, y, 
		method='logistic',
		**kwargs):
		
		if method == 'logistic':
			self.model = LogisticRegressionCV(n_jobs=-1, **kwargs)

		self.model.fit(X, y)

		return

	def predict(self, X_test, 
				probabilities=False,
				**kwargs):
		if probabilities:
			return self.model.predict_proba(X_test)

		return self.model.predict(X_test)


	def select_k(self, train_corpus, validation_corpus,
			y_train, y_validation,
			vocab,
			K_list=np.arange(10, 101, 10),
			**kwargs):
		'''
		K_list: a list of integers for the
		parameter K (number of topics) from
		which an optimal value will be
		selected through validation.

		Documents need to be pre-processed.
		'''
		assert type(train_corpus) != str
		assert type(validation_corpus) != str

		print("The model will select optimal K via validation.")

		# train_set, validation_set = train_test_split(data, test_size=validation_size)
		# train_docs = train_set[X_col]
		# validation_docs = validation_set[X_col]
		model_metrics = []

		for num_topics in K_list:
			lda = LDA(K = num_topics, alpha = 0.1, eta = 0.01)
			theta_train, beta, elbo = lda.fit(train_corpus, vocab, verbose=False, **kwargs)
			theta_validate, logl = lda.infer(validation_corpus)

			self.fit(theta_train, y_train)

			prediction = self.predict(theta_validate)
			accuracy = accuracy_score(y_validation, prediction)

			model_metrics.append((num_topics, accuracy))

		self._Ks = model_metrics
		

		best_K, best_accuracy = max(model_metrics,key=itemgetter(1))
		self.best_K = best_K
		print('The best model has %d topics and achieves %.2f accuracy' %(best_K, best_accuracy))

		return best_K

class BinaryGenerativeClassifier(object):
	'''
	A generative classifier that takes in two
	model instances and classifier an observation
	according to the more likely model.
	'''

	def __init__(self, model_1, model_2):
		print('A generative classifier is initialised.')
		self.model_1 = model_1
		self.model_2 = model_2

	def predict_proba(self, test_corpus, priors):
		'''
		test_doc: a list of two lists,
		term ids and term counts of the
		document.
		'''
		theta_test_g, elbo_g = self.model_1.infer(test_corpus)
		theta_test_rec, elbo_rec = self.model_2.infer(test_corpus)

		assert type(test_corpus) == list
		assert len(priors) == 2

		doc_term_ids = test_corpus[0]
		doc_term_counts = test_corpus[1]

		assert len(doc_term_ids) == len(doc_term_counts)
		num_docs = len(doc_term_ids)

		doc_probs = []
		for d in range(num_docs):
		    one_doc_term_ids = doc_term_ids[d]
		    one_doc_term_counts = doc_term_counts[d]

		    theta_doc_model_1 = theta_test_g[d, :]
		    beta_doc_model_1 = beta_g[:, one_doc_term_ids]
		    words_prob_model_1 = np.dot(theta_doc_model_1, beta_doc_model_1) * one_doc_term_counts
		    # log[P(model 1 | doc)]
		    logp_model_1 = np.sum(np.log(words_prob_model_1)) + np.log(priors[0])

		    theta_doc_model_2 = theta_test_rec[d, :]
		    beta_doc_model_2 = beta_rec[:, one_doc_term_ids]
		    words_prob_model_2 = np.dot(theta_doc_model_2, beta_doc_model_2) * one_doc_term_counts
		    # log[P(model 2 | doc)]
		    logp_model_2 = np.sum(np.log(words_prob_model_2)) + np.log(priors[1])

		    log_probs = np.array([logp_model_1, logp_model_2])

		    probabilities = np.exp(log_probs - logsumexp(log_probs))

		    doc_probs.append(probabilities)

		return doc_probs

	def predict(self, test_corpus, priors):

		return list(map(lambda x: 0 if np.argmax(x) == 0 else 1), self.predict_proba(test_corpus, priors))










# class Evaluation(object):

# 	def print_conf_matrix(self):

# 	def print_metrics(self):




