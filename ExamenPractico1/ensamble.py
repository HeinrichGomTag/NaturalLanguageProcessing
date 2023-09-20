import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# --- PREPROCESAMIENTO DE DATOS ---

# Carga del modelo de lenguaje español de SpaCy para lematización
nlp = spacy.load('es_core_news_sm')

def lemmatize_text(text):
    """Lematiza el texto proporcionado."""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def count_punctuation(text):
    """Cuenta los signos de puntuación en el texto."""
    return sum(1 for char in text if char in '.,;:!?')

def avg_word_length(text):
    """Calcula la longitud promedio de las palabras en el texto."""
    words = text.split()
    return sum(len(word) for word in words) / len(words)

# Carga y preprocesamiento del conjunto de datos
df2 = pd.read_csv('DataSet para entrenamiento del modelo.csv')
df2['title'] = df2['title'].apply(lemmatize_text)
df2['title_length'] = df2['title'].apply(len)
df2['word_count'] = df2['title'].apply(lambda x: len(x.split()))
df2['punctuation_count'] = df2['title'].apply(count_punctuation)
df2['avg_word_length'] = df2['title'].apply(avg_word_length)

# División del conjunto de datos en entrenamiento y prueba
X = df2[['title', 'title_length', 'word_count', 'punctuation_count', 'avg_word_length']]
y = df2['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Vectorización del texto
vectorizer = TfidfVectorizer(lowercase=True, stop_words=['de', 'para'])
X_train_vectorized = vectorizer.fit_transform(X_train['title'])
X_test_vectorized = vectorizer.transform(X_test['title'])

# Incorporación de características adicionales
X_train_vectorized = np.hstack((X_train_vectorized.toarray(), X_train[['title_length', 'word_count', 'punctuation_count', 'avg_word_length']].values))
X_test_vectorized = np.hstack((X_test_vectorized.toarray(), X_test[['title_length', 'word_count', 'punctuation_count', 'avg_word_length']].values))

# Balanceo de clases con SMOTE
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vectorized, y_train)

# --- ENTRENAMIENTO DE MODELOS Y OPTIMIZACIÓN ---

# Optimización de parámetros para LinearSVC
param_grid_svc = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search_svc = GridSearchCV(LinearSVC(dual=False, max_iter=5000), param_grid_svc, cv=StratifiedKFold(n_splits=5), n_jobs=-1)
grid_search_svc.fit(X_train_resampled, y_train_resampled)

# Optimización de parámetros para SGDClassifier
param_grid_sgd = {'alpha': [0.0001, 0.001, 0.01, 0.1], 'penalty': ['l1', 'l2']}
grid_search_sgd = GridSearchCV(SGDClassifier(), param_grid_sgd, cv=StratifiedKFold(n_splits=5), n_jobs=-1)
grid_search_sgd.fit(X_train_resampled, y_train_resampled)

# Optimización de parámetros para KNeighborsClassifier
param_grid_knn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=StratifiedKFold(n_splits=5), n_jobs=-1)
grid_search_knn.fit(X_train_resampled, y_train_resampled)

# # Modelo de ensemble con votación suave
# eclf = VotingClassifier(estimators=[('svc', grid_search_svc.best_estimator_), 
#                                     ('sgd', grid_search_sgd.best_estimator_), 
#                                     ('knn', grid_search_knn.best_estimator_)], 
#                         voting='soft')
# # Entrenamiento y evaluación del modelo de ensemble
# eclf.fit(X_train_resampled, y_train_resampled)
# y_pred_ensemble = eclf.predict(X_test_vectorized)
# accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
# print(f"Accuracy del modelo de ensemble: {accuracy_ensemble}")

# # --- EVALUACIÓN DE MODELOS INDIVIDUALES ---

# Evaluación de LinearSVC
y_pred_svc = grid_search_svc.best_estimator_.predict(X_test_vectorized)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print(f"Accuracy del modelo LinearSVC optimizado: {accuracy_svc}")

# Evaluación de SGDClassifier
# y_pred_sgd = grid_search_sgd.best_estimator_.predict(X_test_vectorized)
# accuracy_sgd = accuracy_score(y_test, y_pred_sgd)
# print(f"Accuracy del modelo SGDClassifier optimizado: {accuracy_sgd}")

# Evaluación de KNeighborsClassifier
# y_pred_knn = grid_search_knn.best_estimator_.predict(X_test_vectorized)
# accuracy_knn = accuracy_score(y_test, y_pred_knn)
# print(f"Accuracy del modelo KNeighborsClassifier optimizado: {accuracy_knn}")

# # --- EVALUACIÓN DE MODELO ENSAMBLADO ---

# y_pred_ensemble = eclf.predict(X_test_vectorized)
# accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
# print(f"Accuracy del modelo de ensemble: {accuracy_ensemble}")

# # OUTPUT

