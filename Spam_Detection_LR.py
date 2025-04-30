import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
data = pd.read_csv("spam.csv",encoding = 'latin-1')[['v1','v2']]
data.columns = ['label','text']
#print(data.head())
#print(data.isnull().sum())

data['label']=data['label'].map({'ham':0,'spam':1})
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):

    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]

    return ' '.join(words)

data['cleaned_text'] = data['text'].apply(preprocess_text)

tfidf = TfidfVectorizer(max_features = 5000)
X = tfidf.fit_transform(data['cleaned_text'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression(class_weight='balanced',max_iter=1000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))