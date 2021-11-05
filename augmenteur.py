from nltk.tokenize import word_tokenize
import random
import csv
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

def augment(sentence,n):
    new_sentences = []
    words = word_tokenize(sentence)
    for i in range(n):
        random.shuffle(words)
        new_sentences.append(' '.join(words))
    new_sentences = list(set(new_sentences))
    return new_sentences

stemmer = FrenchStemmer()





sw = stopwords.words('French')

ponctu =  ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '’']



resp1 =  augment('ne saurait donc être tenue pour responsable d’un quelconque dommage que tout pourrait subir à la suite d’une telle utilisation.', 200)

resp2 = augment('En aucun cas nous n’encourons de responsabilité pour dommages indirects ou directs', 100) 

resp3 =  augment ('décline expressément en vertu des présentes l’ensemble des garanties', 100)

modif1 = augment('Nous nous réservons le droit de faire des changements à nos termes et conditions, y compris les présentes Conditions Générales de Vente à tout moment.', 200000)

modif2 = augment ('Nous nous réservons le droit de modifier nos tarifs et modalités en respectant un préavis jours minimum.', 100)

modif3 = augment ('nous pouvons être amenés à modifier unilatéralement ou remplacer certaines stipulations.', 100)

resil1 = augment ('se réserve le droit, de manière discrétionnaire, de suspendre ou résilier à tout moment et sans préavis', 200)

resil2 = augment ('se réserve la possibilité de résilier le contrat en cas d’inexécution par des conditions générales ou particulières', 100)

resil3 = augment ('L’abonnement le contrat les présentes  résilié de plein droit sans préavis, ni indemnité', 100)


liste_phrases = []

liste_label = []

dico ={'clauses': liste_phrases, 'label' : liste_label}



liste_phrases.extend(resp1)
liste_phrases.extend(resp2)
liste_phrases.extend(resp3)

for el in liste_phrases:
	liste_label.append('1')
	




liste_phrases.extend(modif1)
liste_phrases.extend(modif2)
liste_phrases.extend(modif3)


for el in zip (modif1):
	liste_label.append('0')

for el in zip (modif2):
	liste_label.append('0')

for el in zip (modif3):
	liste_label.append('0')

	

	
liste_phrases.extend(resil1)
liste_phrases.extend(resil2)
liste_phrases.extend(resil3)


for el in zip (resil1):
	liste_label.append('2')

for el in zip (resil2):
	liste_label.append('2')

for el in zip (resil3):
	liste_label.append('2')


with open('test-phrases-2.csv') as f:
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




