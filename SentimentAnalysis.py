from flask import Flask, request, Response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes 
import pandas as pd
import numpy as np 
import json 

print("Starting machine training.")

fields = ['ItemID', 'Sentiment', 'SentimentText'] 
df = pd.read_csv("CleanSentiment.csv", encoding='latin-1', usecols=fields)
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii')
x = vectorizer.fit_transform(df['SentimentText'].values.astype('U'))
y = list(df.Sentiment)
clf = naive_bayes.MultinomialNB()
clf.fit(x, y)

test = np.array([
"I think that item really sucks", 
"You are bad at this game, please leave",
"You stupid moron"
])
test_vector = vectorizer.transform(test.astype('U'))
print(clf.predict(test_vector))

print("Finished machine training.")

app = Flask(__name__)

@app.route('/', methods=['POST'])
def RunSentimentAnalysis():
	if not request.get_json():
		abort(400)
		
	sentences = np.array(request.get_json())
	test_vector = vectorizer.transform(sentences.astype('U'))
	prediction = clf.predict(test_vector)
	predList = prediction.tolist() 
	return Response(json.dumps(predList), mimetype='application/json')
	
if (__name__ == '__main__'):
	app.run(port=5000)