from sklearn.feature_extraction.text import CountVectorizer

corpus = ['Relay the information',
          'I do not want to be in middle of this right now',
          'Sometimes things happen for a reason',
          'What do you mean by that?']

# Unigrams
vectorizer = CountVectorizer()
x1 = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(x1.toarray())

# Bigrams
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2,2))
x2 = vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names())
print(x2.toarray())

# Unigrams and Bigrams
vectorizer3 = CountVectorizer(analyzer='word', ngram_range=(1,2))
x3 = vectorizer3.fit_transform(corpus)
print(vectorizer3.get_feature_names())
print(x3.toarray())

vectorizer4 = CountVectorizer(stop_words=['and','the','is','and'], binary=False) #Binary=False Frequency Table
x4 = vectorizer4.fit_transform(corpus)
print(vectorizer4.get_feature_names())
print(x4.toarray())

vectorizer5 = CountVectorizer(stop_words='english', binary=True)
x5 = vectorizer5.fit_transform(corpus)
print(vectorizer5.get_feature_names())
print(x5.toarray())