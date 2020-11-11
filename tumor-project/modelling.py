# Loading packages
import pandas as pd
import nltk
import re
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import LatentDirichletAllocation


# Reading in data
raw_dat = pd.read_csv('data/training_text', sep=r"\|\|", engine='python',
                      skiprows=1, names=['id', 'text']).set_index('id')
targets = pd.read_csv('data/training_variants')

# Creating some simple features
raw_dat['word_count'] = raw_dat['text'].apply(lambda x: len(str(x).split(" ")))
raw_text = np.array(raw_dat['text'])

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
stop_words += ["),", ").", "-", ");", ",."]


def normalize_doc(doc):
    # lower case and remove special characters\whitespaces
    doc = str(doc)
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words if token not in set(string.punctuation)]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc


# Applying cleaning function
normalize_document = np.vectorize(normalize_doc)
norm_text = normalize_document(raw_text)

# Need to fix the way we are pre-processing the data.
x_train, x_TEST, y_train, y_TEST = train_test_split(
    norm_text, targets['Class'], random_state=0, train_size=.95, stratify=targets['Class'])
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=.8, random_state=0)


def get_x_datasets(cv, x_train1, x_test1, x_TEST1):
    cv_matrix = cv.fit_transform(x_train1)
    cv_matrix = cv_matrix.toarray()
    vocab = cv.get_feature_names()
    x_train = pd.DataFrame(cv_matrix, columns=vocab)

    cv_matrix_test = cv.transform(x_test1)
    cv_matrix_test = cv_matrix_test.toarray()
    vocab_test = cv.get_feature_names()
    x_test = pd.DataFrame(cv_matrix_test, columns=vocab_test)

    cv_matrix_TEST = cv.transform(x_TEST1)
    cv_matrix_TEST = cv_matrix_TEST.toarray()
    vocab_TEST = cv.get_feature_names()
    x_TEST = pd.DataFrame(cv_matrix_TEST, columns=vocab_TEST)

    return x_train, x_test, x_TEST, vocab


# Creating features baby
lda = LatentDirichletAllocation(n_components=9)
lda.fit([x_train])

cv = CountVectorizer(max_features=1000)
x_train, x_test, x_TEST, vocab = get_x_datasets(cv, x_train, x_test, x_TEST)

gbm = GradientBoostingClassifier()
rf = RandomForestClassifier()
nb = GaussianNB()
knn = KNeighborsClassifier()
clfs = [gbm, rf, nb, knn]

feature_count = [5, 10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000]
feature_count2 = [3000, 4000, 5000, 6000, 7000]
scores = []
scores_bigram = []
scores_mixed = []

for clf in clfs:
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    scores = accuracy_score(y_test, predictions)
    print('{} - {}'.format(clf.__class__.__name__, scores))

for num in feature_count:
    cv = CountVectorizer(max_features=num, ngram_range=(1, 2))
    x_train1, x_test1, x_TEST1, vocab = get_x_datasets(cv, x_train, x_test, x_TEST)
    gbm.fit(x_train1, y_train)
    predictions = gbm.predict(x_test1)
    scores_mixed.append(accuracy_score(y_test, predictions))
    print(scores_mixed)

feature_trend = pd.DataFrame({'Unigram': scores, 'Bigram': scores_bigram,
                              'Unigram & Bigram': scores_mixed}, feature_count)
feature_trend.to_csv('feature_trend.csv')

# Tuning the model
cv = CountVectorizer(max_features=500, ngram_range=(1, 2))
x_train1, x_test1, x_TEST1, vocab = get_x_datasets(cv, x_train, x_test, x_TEST)
gbm.fit(x_train1, y_train)
predictions = gbm.predict(x_test1)
accuracy_score(y_test, predictions)

# Hyper parameter tuning
n_estimators = [int(x) for x in np.linspace(100, 2000)]
learning_rate = [int(x) for x in np.linspace(1, 3)]
max_depth = [int(x) for x in np.linspace(1, 5)]
random_grid = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth}

gbm_random = RandomizedSearchCV(estimator=gbm, param_distributions=random_grid, n_iter=5, random_state=0, n_jobs=-1)
gbm_random.fit(x_train1, y_train)

# Accuracy by features and n-gram
# Most common words

gbm_final = GradientBoostingClassifier(n_estimators=200, learning_rate=.09)
gbm_final.fit(x_train1, y_train)
predictions = gbm_final.predict(x_test1)
accuracy_score(y_test, predictions)

predictions_FINAL = gbm_final.predict(x_TEST1)
accuracy_score(y_TEST, predictions_FINAL)


feature_importances = pd.DataFrame(gbm_final.feature_importances_, index=x_train1.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

feature_importances.to_csv('feature_importances.csv')

pd.DataFrame(x_train1.sum()).to_csv('wordcounts.csv')
