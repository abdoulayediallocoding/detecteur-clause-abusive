import csv
import random
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from nltk.stem.snowball import FrenchStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

stemmer = FrenchStemmer()





sw = stopwords.words('French')

ponctu =  ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '’']

liste_phrases = []

liste_label = []

dico ={'clauses': liste_phrases, 'label' : liste_label}


with open('test-phrases-bonnes.csv') as f:
	reader = csv.reader(f)
	
	for r in reader:
			liste_phrases.append(r[0])
			liste_label.append(r[1])
		
	


Corpus = pd.DataFrame(dico)


	
	
	
	
Corpus['clauses'].dropna(inplace=True)

Corpus['clauses'] = [entry.lower() for entry in Corpus['clauses']]

Corpus['clauses']= [word_tokenize(entry) for entry in Corpus['clauses']]


for phrase in Corpus['clauses']:
	for mot in phrase: 
		if (mot) in ponctu or len(mot) < 2 or mot in sw :
			phrase.remove(mot)
		else:
			stemmer.stem (mot)

Corpus['clauses'] = str(Corpus['clauses'])


X, Y = Corpus['clauses'], Corpus['label']

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y)

vectorizer = CountVectorizer()

counts = vectorizer.fit_transform (X_train)


classifier = MultinomialNB()
targets = Y_train
classifier.fit(counts, targets)
counts_test = vectorizer.transform (X_test)
predictions_NB = classifier.predict(counts_test)
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Y_test)*100)



SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
SVM.fit(counts,targets)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(counts_test)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Y_test)*100)




clf = LogisticRegression(random_state=0).fit(counts, targets)
predictions_clf = clf.predict(counts_test)
print("clf Accuracy Score -> ",accuracy_score(predictions_clf, Y_test)*100)

