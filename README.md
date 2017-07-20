# LDA

An implementation of latent Dirichlet allocation with variational inference, hyperparameter optimization, selection of K through cross-validation. The algorithm was used to predict US recessions from monetary policy texts. My research shows the algorithm is predictive up to a quarter and accuracy can be as high as 90% or more, if we augment topic features with macroeconomic indices.

Thesis can be found [here][https://dkn22.github.io/ds/thesis.pdf].


- preprocess.py: a Parser class to pre-process text
  - minutes_ngram_map.py: a dictionary of mappings from n-grams to unigrams for common economic phrases
- topicmodel.py: an LDA class for mean-field variational inference
- classifier.py: classes for discriminative classifier (via sklearn Logistic Regression) and generative classifier (based on LDA topic model from topicmodel.py)
- evaluation.py: various utility functions and functions to compute/evaluate models based on the area-under-the-curve (AUC) and associated asymptotically normal hypothesis testing
- plotter.py: functions to visualize classification performance of a model (ROC curve, confusion matrix, etc.)



The data used to produce visualizations and classification output can be found in train_df.csv and test_df.csv. 



The stored latent variables can be found in the "Stored Latents" folder; these were used to obtain the *exact* results reported in the Jupyter notebooks.
