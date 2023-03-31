from json import load
import nltk
import random, pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
#import data_processing as dt

lemmatizer=WordNetLemmatizer()

class Testing:
    def __init__(self):
        #load the intent file
        #self.dictConv =load(open('conversations',"rb"))
        self.dictConv = load(open('data.json', 'r'))
        #load the training_data file which contains training data
        data=pickle.load(open('training_data',"rb"))
        self.words=data['words']
        self.classes=data['classes']
        self.model=load_model('chatbot_model.h5')
        #set the error threshold value
        self.ERROR_THRESHOLD=0.2
        self.ignore_words=list("!@#$%^&*?")

    def clean_up_sentence(self,sentence):
        #tokenize each sentence (user's query)
        sentence_words=word_tokenize(sentence.lower())
        #lemmatize the word to root word and filter symbols words
        sentence_words=list(map(lemmatizer.lemmatize,sentence_words))
        sentence_words=list(filter(lambda x:x not in self.ignore_words,sentence_words))
        return set(sentence_words)

    def wordvector(self,sentence):
        #initialize CountVectorizer
        #txt.split helps to tokenize single character
        cv=CountVectorizer(tokenizer=lambda txt: txt.split())
        sentence_words=' '.join(self.clean_up_sentence(sentence))
        words=' '.join(self.words)

        #fit the words into cv and transform into one-hot encoded vector
        vectorize=cv.fit([words])
        word_vector=vectorize.transform([sentence_words]).toarray().tolist()[0]
        return(np.array(word_vector))

    def classify(self,sentence):
        #predict to which class(tag) user's query belongs to
        results=self.model.predict(np.array([self.wordvector(sentence)]))[0]
        #store the class name and probability of that class
        results = list(map(lambda x: [x[0],x[1]], enumerate(results)))
        #accept those class probability which are greater then threshold value,0.5
        results = list(filter(lambda x: x[1]>self.ERROR_THRESHOLD ,results))

        #sort class probability value in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []

        for i in results:
            return_list.append((self.classes[i[0]],str(i[1])))
        return return_list

    def results(self,sentence):
        #if context is maintained then filter class(tag) accordingly
        return self.classify(sentence)

    def response(self,sentence):
        #get class of users query
        results=self.results(sentence)
        print(sentence,results)
        ans=""
        if results:
            for item in self.dictConv:
                if item["tag"]==results[0][0]:
                    ans= item["responses"]
                    break

        if ans!="":
            return ans
        else:
            ans = "Sorry ! I am still Learning.\nYou can train me by providing more datas."
            return ans