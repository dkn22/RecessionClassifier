# LDA

An implementation of latent Dirichlet allocation with variational inference, hyperparameter optimization, selection of K through cross-validation.



- preprocess.py: a Parser class to pre-process text
  - minutes_ngram_map.py: a dictionary of mappings from n-grams to unigrams for common economic phrases
- topicmodel.py: an LDA class for mean-field variational inference
- classifier.py: classes for discriminative classifier (via sklearn Logistic Regression) and generative classifier (based on LDA topic model from topicmodel.py)
- evaluation.py: various utility functions and functions to compute/evaluate models based on the area-under-the-curve (AUC) and associated asymptotically normal hypothesis testing
- ​