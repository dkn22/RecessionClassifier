from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from minutes_ngram_map import econ_dict
import re

class Parser(object):
	def __init__(self, lemmatise=False, stem=True, 
					replace_ngrams=True):

		self._lemmatise = lemmatise
		self._stem = stem
		self._replace_ngrams = replace_ngrams

	def parse_vocab(self, raw_docs, custom_stop_words=None,
					replace_dict=None):
		'''
		raw_docs: a list-like object containing documents as raw strings.
		'''
		# vectorizer = CountVectorizer(stop_words='english', **kwargs)
		# dtm = vectorizer.fit_transform(df['text'])
		# self.stop_words = vectorizer.stop_words_
		# self.vocab = dict((value, key) for key, value in vectorizer.vocabulary_.items())

		if type(raw_docs) == str:
			raw_docs = [raw_docs]

		punctuation = re.compile(r'[-.?!,":;()|0-9]')

		try:
			stop_words = set(open('stopwords_long.txt').read().splitlines())
		except Exception:
			stop_words = set(stopwords.words('english'))

		stop_words.update(['.', ',', '"', "'", '?', '!', 
			':', ';', '(', ')', '[', ']', '&',
			'{', '}', '..', '...', '--', '', '/', "''",
			"/'s", "'s", "$", "``", "`", "c", "s"])
		stop_words.update(['january', 'february',
			'march', 'april', 'may', 'june',
			'july', 'august', 'september',
			'october', 'november', 'december',
			'first', 'second', 'third', 'fourth',
			'percent', 'per cent']
			)

		if custom_stop_words is not None and type(custom_stop_words) == 'list':
			stop_words.update(custom_stop_words)

		self.stop_words = stop_words

		lemmatiser = WordNetLemmatizer()
		p_stemmer = PorterStemmer()
		docs = []
		for doc in raw_docs:

			# word_list = re.split('\s+', doc.lower())
			raw = doc.lower()
			if self._replace_ngrams:
				raw = self.replace_ngrams(raw, replace_dict)

			token_list = word_tokenize(raw)
			token_list = [word for word in token_list if word not in stop_words]
			words = (punctuation.sub("", word).strip() for word in token_list if not word.startswith('m'))
			tokens = [word for word in words if word not in stop_words]

			if self._lemmatise:
				tokens = [lemmatiser.lemmatize(t) for t in tokens] # lemmatisation of tokens
			if self._stem:
				tokens = [p_stemmer.stem(i) for i in tokens]
			docs.append(tokens)
		vocab = Dictionary(docs)
		# word_ids = vocab.keys()
		# print('Created the vocabulary. # terms:', len(vocab.keys()))

		return docs, vocab

	@staticmethod
	def remove_low_tfidf_tokens(docs, vocab, cutoff=0.025):
		'''
		docs: processed documents as lists of tokens
		vocab: instance of gensim Dictionary

		Removes tokens with a TF-IDF score below cutoff.
		'''
		corpus = [vocab.doc2bow(text) for text in docs]
		tfidf = TfidfModel(corpus, id2word=vocab)

		low_value_words = []
		for bow in corpus:
			low_value_words += [id for id, value in tfidf[bow] if value < cutoff]

		vocab.filter_tokens(bad_ids=low_value_words)
		print('Trimmed the vocabulary. # terms:', len(vocab.keys()))

		return vocab

	@staticmethod
	def replace_ngrams(string, replace_dict=None):
		'''
		replace_dict: dictionary of mappings.

		The default replace_dict was constructed
		based on the bi-grams and tri-grams
		which had frequency >100 in the 
		Federal Reserve Minutes data and were
		economically meaningful.
		'''
		assert type(string) == str

		if replace_dict is None:
			replace_dict = econ_dict

		for k, v in replace_dict.items():
			string = string.replace(" " + k + " ", " " + v + " ")

		return string

	def parse_corpus(self, docs, vocab):
		'''
		docs: a list of documents, where each document is a list of strings (words).
		vocab: an instance of gensim Dictionary.
		'''
		corpus = [vocab.doc2bow(text) for text in docs]

		doc_term_ids = []
		doc_term_counts = []
		for doc in corpus:
			doc_term_ids.append([doc[i][0] for i in range(len(doc))])
			doc_term_counts.append([doc[j][1] for j in range(len(doc))])

		assert len(doc_term_ids) != 0
		assert len(doc_term_ids) == len(doc_term_counts)
		print('Successfully parsed the corpus. # docs: ', len(doc_term_ids))
		print('Vocabulary size, # tokens: ', len(vocab.keys()))

		corpus = (doc_term_ids, doc_term_counts)

		return corpus

class LemmaTokenizer(object):
	'''
	WordNet Lemmatizer

	Lemmatize using WordNet's built-in morphy function.
	Returns the input word unchanged if it cannot be found in WordNet.

	source: http://www.nltk.org/_modules/nltk/stem/wordnet.html
	'''
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]