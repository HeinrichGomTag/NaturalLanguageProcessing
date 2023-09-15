import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Cargar el modelo de lenguaje español de SpaCy para la lematización
nlp = spacy.load('es_core_news_sm')

# Función para lematizar el texto
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# Función para contar signos de puntuación
def count_punctuation(text):
    return sum(1 for char in text if char in '.,;:!?')

# Función para calcular la longitud promedio de las palabras
def avg_word_length(text):
    words = text.split()
    return sum(len(word) for word in words) / len(words)

# Cargar y preprocesar el conjunto de datos
df2 = pd.read_csv('DataSet para entrenamiento del modelo.csv')
df2['title'] = df2['title'].apply(lemmatize_text)
df2['title_length'] = df2['title'].apply(len)
df2['word_count'] = df2['title'].apply(lambda x: len(x.split()))
df2['punctuation_count'] = df2['title'].apply(count_punctuation)
df2['avg_word_length'] = df2['title'].apply(avg_word_length)

# Dividir el conjunto de datos
X = df2[['title', 'title_length', 'word_count', 'punctuation_count', 'avg_word_length']]
y = df2['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Vectorizar el texto
vectorizer = TfidfVectorizer(lowercase=True, stop_words=['de', 'para'])
X_train_vectorized = vectorizer.fit_transform(X_train['title'])
X_test_vectorized = vectorizer.transform(X_test['title'])

# Agregar las características adicionales
X_train_vectorized = np.hstack((X_train_vectorized.toarray(), X_train[['title_length', 'word_count', 'punctuation_count', 'avg_word_length']].values))
X_test_vectorized = np.hstack((X_test_vectorized.toarray(), X_test[['title_length', 'word_count', 'punctuation_count', 'avg_word_length']].values))

# Balancear las clases con SMOTE
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vectorized, y_train)

# Búsqueda en cuadrícula para LinearSVC
param_grid_svc = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100]
}
grid_search_svc = GridSearchCV(LinearSVC(), param_grid_svc, cv=StratifiedKFold(n_splits=5))
grid_search_svc.fit(X_train_resampled, y_train_resampled)
best_clf_svc = grid_search_svc.best_estimator_

# Búsqueda en cuadrícula para SGDClassifier
param_grid_sgd = {
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'penalty': ['l1', 'l2']
}
grid_search_sgd = GridSearchCV(SGDClassifier(), param_grid_sgd, cv=StratifiedKFold(n_splits=5))
grid_search_sgd.fit(X_train_resampled, y_train_resampled)
best_clf_sgd = grid_search_sgd.best_estimator_

# Búsqueda en cuadrícula para KNeighborsClassifier
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=StratifiedKFold(n_splits=5))
grid_search_knn.fit(X_train_resampled, y_train_resampled)
best_clf_knn = grid_search_knn.best_estimator_

# Ensemble con votación suave
eclf = VotingClassifier(estimators=[('svc', best_clf_svc), ('sgd', best_clf_sgd), ('knn', best_clf_knn)], voting='soft')
eclf.fit(X_train_resampled, y_train_resampled)
y_pred = eclf.predict(X_test_vectorized)

# Resultados
# print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
