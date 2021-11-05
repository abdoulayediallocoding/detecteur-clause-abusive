import csv
import nltk.data
import pandas as pd
import numpy as np
import sklearn
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from collections import defaultdict
from nltk.corpus import wordnet as wn




#Corpus = pd.read_csv('clauses abusives.csv', encoding ='latin-1')




liste_phrases = []

liste_label = []

dico ={'clauses': liste_phrases, 'label' : liste_label}


with open('clauses abusives csv.csv') as f:
	reader = csv.reader(f)
	
	for r in reader:
		liste_phrases.append(r[0])
		liste_label.append(r[1])
		
	
		
	

Corpus = pd.DataFrame(dico)




# Step - a : Remove blank rows if any.
Corpus['clauses'].dropna(inplace=True)
# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
Corpus['clauses'] = [entry.lower() for entry in Corpus['clauses']]

# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
Corpus['clauses']= [word_tokenize(entry) for entry in Corpus['clauses']]


Corpus['clauses']= str (Corpus['clauses'])
X, Y = Corpus['clauses'], Corpus['label']


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,)

vectorizer = CountVectorizer()

counts = vectorizer.fit_transform (X_train.values)

classifier = MultinomialNB()
targets = Y_train.values

classifier.fit(counts, targets)

counts_test = vectorizer.transform (X_test.values)

predictions_NB = classifier.predict(counts_test)

print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Y_test)*100)

phrase=["Il est possible que nous modifiions le présent Contrat, par exemple dans le but de refléter des ajustements apportés à notre Service ou pour des raisons légales, réglementaires ou de sécurité. "]
phrase = [entry.lower() and entry.isalpha() for entry in phrase]
print(phrase)
phrase = [str(phrase)]

phrase = [word_tokenize(entry) for entry in phrase]

phrase = [str(phrase)]


ok = vectorizer.transform(phrase)

print(classifier.predict_proba(ok)[:,0] *100)




SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)


SVM.fit(counts,targets)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(counts_test)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Y_test)*100)


print(SVM.predict_proba(ok)[:,0] *100)


input("ok")
