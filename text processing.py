liste_phrases = []

liste_label = []

dico ={'clauses': liste_phrases, 'label' : liste_label}


with open('clauses abusives.csv') as f:
	reader = csv.reader(f)
	
	for r in reader:
		liste_phrases.append(r[0])
		liste_label.append(r[1])
		
	
		
	

Corpus = pd.DataFrame(dico)

print(Corpus[0])