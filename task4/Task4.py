#imported all libraries required
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import  TfidfVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score

#reading dataset or csv file
data=pd.read_csv("spam.csv",encoding='latin-1')             #reading dataset
#data cleaning and preprocessing
ps=PorterStemmer()                                          # using porter stemmer for base words
wordnet_lemmatizer = WordNetLemmatizer()
ls=[]
for i in range(0,len(data)):
    newdata=re.sub('[^a-zA-Z]',' ',data['Text'][i])          #removing all unnecessary data except a-zA-Z caharcters
    newdata=newdata.lower()                                  #lowering all words to small alphabets
    newdata=newdata.split()                                  #splitting each word

    #newdata=[ps.stem(word) for word in newdata if not word in stopwords.words('english')]
    newdata = [wordnet_lemmatizer.lemmatize(word) for word in newdata if not word in stopwords.words('english')]
    newdata=''.join(newdata)                                 #joining words after stemming to newdata
    ls.append(newdata)
#print("after stemming: \n",ls)
#creating bag of words model
countvec= CountVectorizer(max_features=5000)                 #document matrix with top features about 5000 are used
X=countvec.fit_transform(ls).toarray()
tfidf_Vect = TfidfVectorizer()
X_tfidf = tfidf_Vect.fit_transform(ls).toarray()
#print(countvec)
#print(X)
y=pd.get_dummies(data['Class'])
#print(y)
y=y.iloc[:,1].values
#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
X_train,X_test,y_train,y_test=train_test_split(X_tfidf,y,test_size=0.20,random_state=42)
#training model using NaiveBayes
NB=MultinomialNB().fit(X_train,y_train)                   #using naivebayes classification technique
pred=NB.predict(X_test)
conf=confusion_matrix(y_test,pred)
print("confusion matrix is:",conf)
score=accuracy_score(y_test,pred)                         #calculating accuracy score on test and predicted data
print("accuracy score is:",score)