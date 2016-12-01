from gensim.corpora import Dictionary
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def _parse_vocab(raw_docs, custom_stop_words=None, 
				lemmatise=False, stem=True):
	'''
	raw_docs: a list-like object containing documents as raw strings.
	'''
	# vectorizer = CountVectorizer(stop_words='english', **kwargs)
	# dtm = vectorizer.fit_transform(df['text'])
	# self.stop_words = vectorizer.stop_words_
	# self.vocab = dict((value, key) for key, value in vectorizer.vocabulary_.items())

	stop_words = set(stopwords.words('english'))
	stop_words.update(['.', ',', '"', "'", '?', '!', 
		':', ';', '(', ')', '[', ']', 
		'{', '}', '..', '...',
		"'s"])
	if custom_stop_words is not None and type(custom_stop_words) == 'list':
		stop_words.update(custom_stop_words)

	lemmatiser = WordNetLemmatizer()
	p_stemmer = PorterStemmer()
	docs = []
	for doc in raw_docs:
		raw = doc.lower()
		if lemmatise:
			tokens = [lemmatiser.lemmatize(t) for t in word_tokenize(raw)] # lemmatisation of tokens
		else:
			tokens = word_tokenize(raw)
			stopped_tokens = [i for i in tokens if i not in stop_words]
		if stem:
			stopped_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
		docs.append(stopped_tokens)
	vocab = Dictionary(docs)
	# word_ids = vocab.keys()
	# print('Created the vocabulary. # terms:', len(vocab.keys()))

	return docs, vocab


def _parse_corpus(docs, vocab):
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
	# print('Successfully parsed the corpus. # docs: %d', %(len(doc_term_ids)))

	return doc_term_ids, doc_term_counts

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