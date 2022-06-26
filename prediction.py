
import pickle
import nltk
from nltk import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer


stemer=PorterStemmer()
token=WordPunctTokenizer()

loaded_model = pickle.load(open('sentiment_modelNB.pkl', 'rb'))

stopwords=set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(stemer.stem(i))

    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))

# input the message
message=input()
message=transform_text(message)
modified_text=tfidf.transform([message])

model_prediction=loaded_model.predict(modified_text)[0]

if model_prediction==1:
    print("Nice movie to Watch")
else:
    print("Bad review by audiences")