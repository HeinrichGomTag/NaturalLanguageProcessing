# Importar las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier

# Cargar el modelo de lenguaje español de SpaCy para la lematización y análisis de sentimiento
nlp = spacy.load('es_core_news_sm')

# Función para determinar el sentimiento del título
def get_sentiment(text):
    doc = nlp(text)
    if doc.sentiment > 0:
        return 'positive'
    elif doc.sentiment < 0:
        return 'negative'
    else:
        return 'neutral'

# Paso 01 - Lectura del conjunto de información
df2 = pd.read_csv('DataSet para entrenamiento del modelo.csv')
df2.index = np.arange(1, len(df2) + 1)

# Paso 02 - Verificación y validación de la información
df2 = df2.dropna()
df2 = df2[df2['title'] != ""]

# Feature Engineering
df2['title_length'] = df2['title'].apply(len)
df2['word_count'] = df2['title'].apply(lambda x: len(x.split()))
df2['punctuation_count'] = df2['title'].apply(lambda x: sum(1 for char in x if char in '.,;:!?'))
df2['sentiment'] = df2['title'].apply(get_sentiment)
df2 = pd.get_dummies(df2, columns=['sentiment'])
print("Feature engineering")

# Paso 05 - Construcción del modelo de NLP
X = df2['title']
Y = df2['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
print("Splitted")

# Definición de modelos
clf_logistic = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
clf_naive_bayes = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
clf_svc = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
print("Defined models")
# Parámetros para la búsqueda en cuadrícula
param_grid = {
    'tfidf__max_df': [0.85, 0.9, 0.95],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__C': [0.1, 1, 10]
}

# Búsqueda en cuadrícula para SVC
grid_search_svc = GridSearchCV(clf_svc, param_grid, cv=5)
grid_search_svc.fit(X_train, Y_train)
best_clf_svc = grid_search_svc.best_estimator_
print("GRID SVC")

# Evaluación de modelos
Predicciones_svc = best_clf_svc.predict(X_test)
print("\nAccuracy del modelo SVC optimizado: ")
print(accuracy_score(Y_test, Predicciones_svc))

# Modelo de Ensemble (Votación)
ensemble_model = VotingClassifier(estimators=[
    ('svc', best_clf_svc),
    ('logistic', clf_logistic),
    ('naive_bayes', clf_naive_bayes)
], voting='hard')

ensemble_model.fit(X_train, Y_train)
ensemble_predictions = ensemble_model.predict(X_test)

print("\nAccuracy del modelo de ensemble: ")
print(accuracy_score(Y_test, ensemble_predictions))
