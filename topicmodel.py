'''
(с) Dat Nguyen (dat.nguyen@cantab.net)
'''

import numpy as np
from scipy.special import gammaln, psi, polygamma
from scipy.misc import logsumexp
from scipy import optimize
import time

# import parser
from utils import *
import warnings


class LDA(object):
    '''
    Mean-field variational inference algorithm for Latent Dirichlet Allocation.

    Specifically, the implementation follows the notation in Blei and Lafferty (2009).

    ---
    References:

    Blei, Ng and Jordan (2003), Latent Dirichlet Allocation;
    Blei and Lafferty (2009), Topic Models.
    '''

# 1. Initialization
    def __init__(self, K, alpha=0.1, eta=0.01, optimize_hyperparam=True,
        ):
        # assert len(alpha) == K or len(list(alpha)) == 1;
        # if len(alpha)==K else alpha*np.ones(self.num_topics);
        self.alpha = alpha
        self.num_topics = K
        # prior on Beta
        self.eta = eta
        self.optimize_hyperparam = optimize_hyperparam

    def _initialize(self, doc_term_ids, doc_term_counts, vocab):
        '''
        document parameters
        '''

        # self.alpha = np.array([1/K]*K)
        # self._phi = []
        self.vocab_size = len(vocab)

        # assert len(doc_term_ids) == len(doc_term_counts)
        self.num_docs = len(doc_term_ids)

        # assert len(self.alpha) == self.num_topics
        self.alpha = self.alpha + np.zeros(self.num_topics)
        # assert len(self.eta) == self.vocab_size
        self.eta = self.eta + np.zeros(self.vocab_size)

        # self._gamma = (1/self.num_topics) * np.ones((self.num_docs, self.num_topics))
        self._gamma = np.random.gamma(100., 1. / 100.,
                                       (self.num_docs, self.num_topics))

        self._lambda = np.random.gamma(100., 1. / 100.,
                                       (self.num_topics, self.vocab_size))
        # TBC: 'seed' docs to initialize (p.11)
        # self._lambda = 1/self.vocab_size * np.ones((self.num_topics, self.vocab_size))

# 2. Learning

    def fit(self, corpus, vocab,
            initial_beta = None,
            max_iter=100,
            convergence_thres=1e-3,
            gamma_max_iter=10,
            gamma_convergence_threshold=1e-3,
            alpha_update_interval=5,
            alpha_optimize_method='hybr',
            alpha_convergence_threshold=1e-5,
            alpha_max_iter=100,
            verbose=True):

        # corpus should consist of a list of lists
        # document term ids and term counts for each document
        assert len(corpus) == 2
        assert len(corpus[0]) == len(corpus[1])

        # docs, vocab = parser._parse_vocab(corpus, **kwargs)
        # print('Created the vocabulary. # terms:', len(vocab.keys()))

        # store the vocabulary as a dictionary to access word strings
        self._vocab = vocab

        # doc_term_ids, doc_term_counts = parser._parse_corpus(docs, vocab)
        # print('Successfully parsed the corpus. # docs: ', len(doc_term_ids))

        doc_term_ids = corpus[0]
        doc_term_counts = corpus[1]
        self._initialize(doc_term_ids, doc_term_counts, vocab)

        if initial_beta is not None:
            assert initial_beta.shape == (self.num_topics, self.vocab_size)
            self._lambda = initial_beta

        # Mean-field algorithm (variational EM)
        counter = 1
        elbo = 0
        previous_elbo = elbo + convergence_thres + 1
        clock = time.time()

        if self.optimize_hyperparam and verbose:
            print('The algorithm will update the hyperparameter alpha using numerical optimization (%s method).' %
                  alpha_optimize_method)

        while abs(elbo - previous_elbo) > convergence_thres and counter < max_iter + 1:

            previous_elbo = elbo

            phi_over_all_d, elbo_without_beta = self.e_step(doc_term_ids,
                                                            doc_term_counts,
                                                            gamma_max_iter,
                                                            gamma_convergence_threshold)

            elbo = self.m_step(phi_over_all_d, elbo_without_beta)

            if self.optimize_hyperparam and counter % alpha_update_interval == 0:
                if alpha_optimize_method == 'newton-raphson':
                    self.update_alpha(
                        alpha_convergence_threshold, alpha_max_iter)
                else:
                    self.optimize_alpha(alpha_optimize_method)

            if verbose and (counter == 1 or counter % 10 == 0):
                print('Successfully completed %d iteration(s) in %ds.' %
                      (counter, time.time() - clock))
                print('ELBO =', elbo)
            counter += 1
        print('Learning completed!\nTotal time taken: %ds\nELBO = %f' %
              (time.time() - clock, elbo))

        if np.any(self.alpha < 1e-5):
            warnings.warn('''Possible numerical issues with 
                hyperparameter optimization due to
                inversion of a singular Hessian matrix.
                Try to initialize alpha to different values.''',
                RuntimeWarning)

        self.theta = self._gamma / np.sum(self._gamma, axis=1).reshape(-1, 1)

        self.beta = self._lambda / np.sum(self._lambda, axis=1).reshape(-1, 1)

        return self.theta, self.beta, elbo  # , phi_over_all_d

    def fit_multiple_runs(self, corpus, vocab, n_runs=10,
            **kwargs):
        '''
        Fit several LDA models with random initializations
        and select the best one.
        '''

        best_elbo = -np.inf

        for run in range(n_runs):
            theta, beta, elbo = self.fit(corpus, vocab, **kwargs)

            if elbo > best_elbo:
                best_theta = theta
                best_beta = beta
                best_elbo = elbo

        self.theta = best_theta
        self.beta = best_beta

        return best_theta, best_beta, best_elbo


# 2.1 E-Step: coordinate ascent in gamma and phi

    def e_step(self, doc_term_ids, doc_term_counts,
               gamma_max_iter=10,
               gamma_convergence_threshold=1e-3):
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
        phi_over_all_d = np.zeros((self.num_topics, self.vocab_size))  # KxV
        # phi_over_all_d = np.zeros((self.vocab_size, self.num_topics)) # VxK

        # expectation of log beta under the variational Dirichlet (p. 10)
        # E_log_beta = psi(lambda) - psi(np.sum(lambda, axis=1))
        E_log_beta = psi(self._lambda) - psi(np.sum(self._lambda,
                                                    axis=1)).reshape(-1, 1)  #  KxV matrix

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

                lastgamma = gammad

                # Calculate phi first
                # Log transformation to avoid the exponential function (slow)
                # since gamma[d,k] is the same for all v, we repeat it using
                # np.tile

                # T_d x K matrix, where T_d = total_term_count
                # This is because updates for phi are repeated for the same
                # term (p. 11)
                log_phid = np.tile(psi(self._gamma[d, :]), reps=(
                    total_term_count, 1)) + E_log_beta[:, term_ids].T
                # log_phid = np.tile(psi(self._gamma[d, :]), reps=(total_term_count, 1)) + E_log_beta[term_ids, :];

                # the equation above is only true proportionally (p. 9)
                # since we took logs, we will substract the normalizer (not
                # divide)
                log_phid_normalizer = logsumexp(
                    log_phid, axis=1)  # sum over topics (cols)
                # since for each topic, sum(probs) = 1

                log_phid = log_phid - log_phid_normalizer.reshape(-1, 1)

                # update for gamma (eq. 14)
                # gammad = self.alpha + np.dot(term_counts.reshape(1, -1), np.exp(log_phid))
                # sum over rows to get a 1xK vector
                gammad = self.alpha + \
                    np.array(
                        np.sum(np.exp(log_phid + np.log(term_counts.reshape(-1, 1))), axis=0))

                self._gamma[d, :] = gammad

                assert len(gammad) == self.num_topics

                if np.mean(abs(gammad - lastgamma)) < gamma_convergence_threshold:
                    break

            # self._gamma[d, :] = gammad

            phi_over_all_d[:, term_ids] += np.exp(log_phid + np.log(term_counts.reshape(-1, 1))).T
            # phi_over_all_d[term_ids, :] += np.exp(log_phid + np.log( term_counts.reshape(-1, 1) ) )

            # NOTE: terms involving psi(gamma) get cancelled
            # because gammad = alpha + \sum_n(phi_dn)
            elbo_without_beta += gammaln(np.sum(self.alpha)) - \
                np.sum(gammaln(self.alpha))

            elbo_without_beta += np.sum(gammaln(gammad)) - \
                gammaln(np.sum(gammad))

            # dot product gives summation over N_d
            # afterwards we sum over K to get Elog[q(z)]
            elbo_without_beta -= np.sum(np.dot(term_counts,
                                               (np.exp(log_phid) * log_phid)))

        return phi_over_all_d, elbo_without_beta

# 2.2 M-step: coordinate ascent in lambda

    def m_step(self, phi_over_all_d, elbo_without_beta):
        '''
        update the variational parameter lambda.
        '''

        self._lambda = self.eta + phi_over_all_d
        # self._lambda = self.eta + phi_over_all_d.T

        elbo = elbo_without_beta

        # (-)Elog[q(lambda)]
        elbo += np.sum(np.sum(gammaln(self._lambda), axis=1) -
                       gammaln(np.sum(self._lambda, axis=1)))

        # Note: terms containing psi(eta) get cancelled out
        # Elog[p(w | Z, beta)] also gets cancelled out

        # Elog[p(beta | eta)] is analogous to Elog[p(theta | alpha)]
        # except we need to sum over K, rather than D
        # since Elog[p(beta | eta)] does not depend on K
        # we can simply multiply by K
        elbo += self.num_topics * \
            (gammaln(np.sum(self.eta)) - np.sum(gammaln(self.eta)))

        return elbo

# 2.3 Dirichlet hyperparameter optimisation

    def update_alpha(self, convergence_threshold=1e-5, max_iter=100):
        '''
        update hyperparameter alpha.

                References: 

        Appendix A.2 and A.4.2 in Blei, Ng, Jordan (2003),
        Minka (2000) at https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf

        '''

        alpha_new = self.alpha

        # alpha sufficient statistics
        gradient_gamma_terms = psi(self._gamma) - psi(np.sum(self._gamma, axis=1)).reshape(-1, 1)
        gradient_gamma_terms = np.sum(gradient_gamma_terms, axis=0)  # scalar

        for iterr in range(max_iter):
            alpha_sum = np.sum(self.alpha)
            gradient = self.num_docs * \
                (psi(alpha_sum) - psi(self.alpha)) + gradient_gamma_terms

            # Hessian = diag(h) + 1z1^T
            # h is a k-dimensional vector
            h = -self.num_docs * polygamma(1, self.alpha)
            z = self.num_docs * polygamma(1, alpha_sum)

            # float division
            c = np.sum(gradient / h) / (1. / z + np.sum(1. / h))

            alpha_update = (gradient - c) / h
            step_size = 1

            # decrease the update to avoid numerical instability issues
            while any(self.alpha - step_size * alpha_update < 0):
                step_size = step_size * 0.9

            alpha_new = self.alpha - step_size * alpha_update

            # measure the average update to evaluate the stopping criterion
            # before assigning the updated alpha
            avg_update = np.mean(abs(alpha_new - self.alpha))
            self.alpha = alpha_new

            # stopping rule
            if avg_update < convergence_threshold:
                break

        # return self.alpha
        return

    def optimize_alpha(self, optimize_method):
        alpha = self.alpha
        gamma = self._gamma
        num_docs = self.num_docs

        def alpha_gradient(alpha, gamma, num_docs):
            alpha_sum = np.sum(alpha)
            gradient_gamma_terms = psi(
                gamma) - psi(np.sum(gamma, axis=1)).reshape(-1, 1)
            gradient_gamma_terms = np.sum(
                gradient_gamma_terms, axis=0)  # scalar

            gradient = num_docs * \
                (psi(alpha_sum) - psi(alpha)) + gradient_gamma_terms

            return gradient

        def alpha_hessian(alpha, gamma, num_docs):
            alpha_sum = np.sum(alpha)
            dim = alpha.shape[0]

            hessian = num_docs * np.ones((dim, dim)) * polygamma(1, alpha_sum)

            diagonal = -num_docs*(polygamma(1, alpha) + polygamma(1, alpha_sum))
            diag_idx = np.diag_indices(dim)
            hessian[diag_idx] = diagonal

            return hessian

        alpha_optimal = optimize.root(alpha_gradient, alpha, jac=alpha_hessian, 
        	args=(gamma, num_docs), method=optimize_method)

        self._alpha = alpha_optimal
        self.alpha = alpha_optimal.x

        # return alpha_optimal
        return

# 3. Out-of-sample prediction

    def infer(self, test_corpus,
                gamma_max_iter=10,
                gamma_convergence_threshold=1e-3,
                custom_alpha=None,
                **kwargs):
        '''
        test_corpus: pre-processed corpus of documents,
        where each document consists of two lists: term ids
        and term counts.
            NB: needs to be pre-processed using model vocabulary.

        Out-of-sample prediction of the doc-topic vector and
        log-likelihood for a previously unseen document(s).

        Vocabulary and topic-word matrix remain unaffected.
        '''

        # test_docs = pd.DataFrame(test_docs)
        # num_docs = test_docs.shape[0]
        # if type(test_docs) == str:
        #     test_docs = [test_docs]

        # test_docs, _ = parser._parse_vocab(test_corpus, **kwargs)
        # doc_term_ids, doc_term_counts = parser._parse_corpus(test_docs, self._vocab)
        # # doc_term_ids is a list of lists
        

        assert len(test_corpus[0]) == len(test_corpus[1])
        doc_term_ids = test_corpus[0]
        doc_term_counts = test_corpus[1]
        num_docs = len(doc_term_ids)

        # initialize
        document_log_likelihood = 0
        predicted_gamma = np.zeros((num_docs, self.num_topics))

        if custom_alpha is not None:
            assert len(custom_alpha) == self.num_topics
            alpha = custom_alpha
        else:
            alpha = self.alpha

        # expectation of log beta under the variational Dirichlet
        E_log_beta = psi(self._lambda) - psi(np.sum(self._lambda,
                                                    axis=1)).reshape(-1, 1)
        E_log_beta_normalizer = logsumexp(E_log_beta, axis=1)

        # normalize E_log_beta to obtain probabilities
        E_log_beta = E_log_beta - E_log_beta_normalizer.reshape(-1, 1)

        for d in range(num_docs):

            # Term ids that appear in document d
            term_ids = np.array(doc_term_ids[d], dtype=np.int)

            # Term counts (for each unique term) in document d
            term_counts = np.array(doc_term_counts[d], dtype=np.int)

            # Number of unique terms in document d
            total_term_count = len(term_ids)

            # Total number of words in document d
            total_word_count = np.sum(term_counts, dtype=np.int)

            # initialize doc-topic vector
            gammad = alpha + total_word_count/self.num_topics
            predicted_gamma[d, :] = gammad

            for iterr in range(gamma_max_iter):
                lastgamma = gammad

                log_phid = np.tile(psi(predicted_gamma[d, :]), reps=(
                    total_term_count, 1)) + E_log_beta[:, term_ids].T

                log_phid_normalizer = logsumexp(log_phid, axis=1)

                log_phid = log_phid - log_phid_normalizer.reshape(-1, 1)

                gammad = (alpha + 
                            np.sum(np.exp(log_phid 
                                    + np.log(term_counts.reshape(-1, 1))), axis=0)
                         )

                predicted_gamma[d, :] = gammad

                if np.mean(abs(gammad - lastgamma)) < gamma_convergence_threshold:
                    break


            # expected document log-likelihood under the variational distribution
                # this is the probability of observing words in the out-of-sample document
                # conditional on the topic distributions (beta) and the topic assignments
            # E_q[log p(w_d | z, beta)] (Appendix A.3 - eq. 15)
            document_log_likelihood += np.sum(np.exp(log_phid.T + np.log(term_counts)) 
                                            * E_log_beta[:, term_ids])

        predicted_gamma = predicted_gamma / np.sum(predicted_gamma, axis=1).reshape(-1, 1)

        return predicted_gamma, document_log_likelihood









































