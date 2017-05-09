from scipy.special import psi # digamma function
from scipy.stats import pearsonr

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

def calc_sd_auc(AUC, n0, n1):
    '''
    n0: Number of observations in the negative class.
    n1: Number of observations in the positive class.
    AUC: the area-under-the-curve statistic.

    The function computes the asymptotic standard deviation of
    the AUC statistic, as outlined in Hsieh and Turnbull (1996).
    '''
    
    Q1 = AUC/(2-AUC)
    
    Q2 = 2*(AUC**2) / (1+AUC)
    
    denominator = 1/(n0*n1)
    
    numerator = AUC*(1-AUC) + (n1 - 1)*(Q1 - AUC**2) + (n0 - 1)*(Q2 - AUC**2)
    
    numerator = numerator**0.5
    
    variance = numerator * denominator
    
    return variance**0.5

def correlate_AUCs(AUC_A, AUC_B, pred_proba_A, pred_proba_B, pos_class_idx):
    '''
    AUC_A: AUC of model A.
    AUC_B: AUC of model B.
    pred_proba_A: predicted probabilities for the positive class under model A.
    pred_proba_B: predicted probabilities for the positive class under model B.
    pos_class_idx: array-like of indices when the true class was positive.
    
    The function computes the Pearson correlation coefficient
    between two models' area-under-the-curve statistics.
    Methodology follows Hanley and McNeil (1982) and 
    Jorda and Taylor (2011).
    '''
    
    try:
        # in case the arrays are H20 frames
        pred_proba_A = np.array(pred_proba_A.as_data_frame())
        pred_proba_B = np.array(pred_proba_B.as_data_frame())
    except Exception:
        pass
    pred_proba_A = np.array(pred_proba_A)
    pred_proba_B = np.array(pred_proba_B)
    
    pos_class_scores_A = pred_proba_A[pos_class_idx]
    pos_class_scores_B = pred_proba_B[pos_class_idx]
    corr_pos_scores_A_B = pearsonr(pos_class_scores_A, pos_class_scores_B)[0]
    
    neg_class_scores_A = np.array([element for i, element in enumerate(pred_proba_A) \
                                   if i not in pos_class_idx])
    neg_class_scores_B = np.array([element for i, element in enumerate(pred_proba_B) \
                                   if i not in pos_class_idx])
    corr_neg_scores_A_B = pearsonr(neg_class_scores_A, neg_class_scores_B)[0]
    
    corr_AUC_A_B = 0.5*(corr_pos_scores_A_B + corr_neg_scores_A_B)
    
    return corr_AUC_A_B

def calc_z_diff(AUC_A, AUC_B, sd_A, sd_B, corr_A_B):
    '''
    This function calculates the z-statistic
    for the test of no difference between the AUCs
    of two classification models.
    '''
    
    var_A = sd_A**2
    var_B = sd_B**2
    
    AUC_diff = AUC_A - AUC_B
    
    z = AUC_diff / np.sqrt(var_A + var_B - 2*corr_A_B*sd_A*sd_B)
    
    return z







