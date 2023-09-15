import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from textblob import TextBlob

# Cargar el modelo de lenguaje español de SpaCy para la lematización
nlp = spacy.load('es_core_news_sm')

# Función para lematizar el texto
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# Función para contar signos de puntuación
def count_punctuation(text):
    return sum(1 for char in text if char in '.,;:!?')

# Función para calcular la longitud promedio de las palabras en un texto
def average_word_length(text):
    words = text.split()
    return sum(len(word) for word in words) / len(words)

# Función para obtener el análisis de sentimiento
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Carga de datos
df2 = pd.read_csv('DataSet para entrenamiento del modelo.csv')
df2.index = np.arange(1, len(df2) + 1)

# Feature Engineering
df2['title'] = df2['title'].apply(lemmatize_text)  # Lematización
df2['title_length'] = df2['title'].apply(len)
df2['word_count'] = df2['title'].apply(lambda x: len(x.split()))
df2['punctuation_count'] = df2['title'].apply(count_punctuation)  # Contar signos de puntuación
df2['avg_word_length'] = df2['title'].apply(average_word_length)  # Longitud promedio de palabras
df2['sentiment'] = df2['title'].apply(get_sentiment)  # Análisis de sentimiento

# División de datos
X = df2['title']
Y = df2['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

# Vectorización
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Sobremuestreo con SMOTE
smote = SMOTE()
X_train_smote, Y_train_smote = smote.fit_resample(X_train_vectorized, Y_train)

# Modelos
clf_svc = LinearSVC()
clf_logistic = LogisticRegression(max_iter=1000)
clf_naive_bayes = MultinomialNB()
clf_sgd = SGDClassifier()
clf_knn = KNeighborsClassifier()

# Entrenamiento y evaluación de cada modelo
models = [clf_svc, clf_logistic, clf_naive_bayes, clf_sgd, clf_knn]
model_names = ['LinearSVC', 'Logistic Regression', 'Naive Bayes', 'SGD Classifier', 'K-Nearest Neighbors']

for model, name in zip(models, model_names):
    model.fit(X_train_smote, Y_train_smote)
    predictions = model.predict(X_test_vectorized)
    accuracy = accuracy_score(Y_test, predictions)
    print(f"Accuracy for {name}: {accuracy:.4f}")

# Ensemble
ensemble_model = VotingClassifier(estimators=[
    ('svc', clf_svc),
    ('logistic', clf_logistic),
    ('naive_bayes', clf_naive_bayes),
    ('sgd', clf_sgd),
    ('knn', clf_knn)
], voting='soft')

ensemble_model.fit(X_train_smote, Y_train_smote)
ensemble_predictions = ensemble_model.predict(X_test_vectorized)
ensemble_accuracy = accuracy_score(Y_test, ensemble_predictions)
print(f"Accuracy for Ensemble Model: {ensemble_accuracy:.4f}")
