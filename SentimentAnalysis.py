from flask import Flask, request, Response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes 
import pandas as pd
import numpy as np 
import json 
import time
from threading import Thread

vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii')
clf = naive_bayes.MultinomialNB()

def TrainSentiment():
	print("Starting machine training.")
	fields = ['ItemID', 'Sentiment', 'SentimentText'] 
	df = pd.read_csv("CleanSentiment.csv", encoding='latin-1', usecols=fields)
	x = vectorizer.fit_transform(df['SentimentText'].values.astype('U'))
	y = list(df.Sentiment)
	clf.fit(x, y)
	print("Finished machine training.")

def SentimentAnalysis(sentences):
	test_vector = vectorizer.transform(sentences.astype('U'))
	prediction = clf.predict(test_vector)
	return prediction.tolist() 

app = Flask(__name__)

@app.route('/', methods=['POST'])
def RunSentimentAnalysis():
	if not request.get_json():
		abort(400)
	sentences = np.array(request.get_json())
	return Response(json.dumps(SentimentAnalysis(sentences)), mimetype='application/json')

t = Thread(target=TrainSentiment)
t.start()

if (__name__ == '__main__'):
	app.run(port=5000)