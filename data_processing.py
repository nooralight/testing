import pandas as pd
import random
import pickle
import nltk
nltk.download('punkt');nltk.download('wordnet');nltk.download('omw-1.4')
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import flatten
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers.legacy import SGD
import json

lemmatizer = WordNetLemmatizer()
class Training:
    def __init__(self):
        #read and load the intent file
        self.df = pd.read_csv('dataset.csv',delimiter='+')
        #self.intents=json.loads(data_file)['intents']
        self.ignore_words=list("!@#$%^&*?")
        self.process_data()
        
    def process_data(self):
        #fetch patterns and tokenize them into words
        self.question =self.df.content_question.tolist()
        self.answer = self.df.content_answer.tolist()
        self.words=list(map(word_tokenize,flatten(self.question)))
        #fetch classes i.e. tags and store in documents along with tokenized patterns
        self.classes = []
        for i in range(len(self.words)):
            self.classes.append(str(i))
        self.dictConversation=[]
        tag = 0
        for i in range(len(self.question)):
            self.dictConversation.append({"tag":str(tag),"patterns":self.question[i],"responses":self.answer[i]})
            tag+=1
        
        self.documents=[]
        for i in range(len(self.words)):
            arr= [self.words[i],str(i)]
            self.documents.append(arr)
            
        ##ignore words
        self.real_words= []
        for word in self.words:
            pick=[]
            for solo in word:
                if solo not in self.ignore_words:
                    pick.append(solo)
            self.real_words.append(pick)


        self.real_words=list(map(str.lower,flatten(self.real_words)))
        self.real_words=list(map(lemmatizer.lemmatize,self.real_words))
        self.real_words=sorted(list(set(self.real_words)))
    
    def getDictConv(self):
        with open('data.json', 'w') as fp:
            json.dump(self.dictConversation, fp)
        
    
    def train_data(self):
        #training the model
        cv=CountVectorizer(tokenizer=lambda txt: txt.split(),analyzer="word",stop_words=None)
        training=[]
        for doc in self.documents:
            #lower case and lemmatize the pattern words
            pattern_words=list(map(str.lower,doc[0]))
            ##print(pattern_words)
            pattern_words=' '.join(list(map(lemmatizer.lemmatize,pattern_words)))
            ##print(pattern_words)
            #train or fit the vectorizer with all words
            #and transform into one-hot encoded vector
            vectorize=cv.fit([' '.join(self.real_words)])
            word_vector=vectorize.transform([pattern_words]).toarray().tolist()[0]

            #create output for the respective input
            #output size will be equal to total numbers of classes
            output_row=[0]*len(self.classes)

            #if the pattern is from current class put 1 in list else 0
            output_row[self.classes.index(doc[1])]=1
            cvop=cv.fit([' '.join(self.classes)])
            out_p=cvop.transform([doc[1]]).toarray().tolist()[0]
            #store vectorized word list long with its class
            training.append([word_vector,output_row])
        random.shuffle(training)
        training=np.array(training,dtype=object)
        train_x=list(training[:,0])#patterns
        train_y=list(training[:,1])#classes
        return train_x,train_y


    def build(self):
        #load the data from train_data function
        train_x,train_y = self.train_data()
        model=Sequential()
        #input layer with latent dimension of 128 neurons and ReLU activation function
        model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
        model.add(Dropout(0.5)) #Dropout to avoid overfitting
        #second layer with the latent dimension of 64 neurons
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.5))
        #fully connected output layer with softmax activation function
        model.add(Dense(len(train_y[0]),activation='softmax'))
        '''Compile model with Stochastic Gradient Descent with learning rate  and
           nesterov accelerated gradient descent'''
        sgd=SGD(learning_rate=1e-2,decay=1e-6,momentum=0.9,nesterov=True)
        model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        #fit the model with training input and output sets
        hist=model.fit(np.array(train_x),np.array(train_y),epochs=5000,batch_size=128,verbose=1)
        #save model and words,classes which can be used for prediction.
        model.save('chatbot_model.h5',hist)
        pickle.dump({'words':self.real_words,'classes':self.classes,'train_x':train_x,'train_y':train_y},open('training_data',"wb"))

Training().getDictConv()
Training().build()
        