import pandas as pd
import spacy
import nltk
import random
from textblob import TextBlob
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Descargar recursos de nltk
nltk.download('wordnet')
nltk.download('stopwords')

nlp = spacy.load('en_core_web_lg')

# Data Augmentation
def get_synonyms(word):
    from nltk.corpus import wordnet
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def replace_with_synonym(sentence):
    words = sentence.split()
    for i, word in enumerate(words):
        synonyms = get_synonyms(word)
        if synonyms:
            words[i] = random.choice(synonyms).replace("_", " ")
    return ' '.join(words)

def augment(df):
    df_augmented = df.copy()
    clickbait_rows = df_augmented[df_augmented["label"] == "clickbait"].copy()
    clickbait_rows["title"] = clickbait_rows["title"].apply(replace_with_synonym)
    return pd.concat([df, clickbait_rows], ignore_index=True)

# Preprocesamiento
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    stopwords = nltk.corpus.stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in stopwords])
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

# Feature Engineering
def feature_engineering(df):
    df['length'] = df['title'].apply(len)
    df['unique_words'] = df['title'].apply(lambda x: len(set(x.split())))
    df['numbers_count'] = df['title'].apply(lambda x: sum(c.isdigit() for c in x))
    df['exclamation_count'] = df['title'].apply(lambda x: x.count('!'))
    df['sentiment'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
    keywords = ["top", "best", "first", "most", "amazing", "incredible"]
    for keyword in keywords:
        df[f'keyword_{keyword}'] = df['title'].apply(lambda x: x.split().count(keyword))
    return df

# Cargar el conjunto de datos
df = pd.read_csv('DataSet para entrenamiento del modelo.csv').dropna()
df = df[df['title'] != ""]

# Punto de interrupción 1: Después de cargar el conjunto de datos# ... [El código anterior]
print("Shape of the original dataset:", df.shape)

# Check imbalance and augment data if necessary
label_counts = df['label'].value_counts()
majority_count = label_counts.max()
minority_count = label_counts.min()

while minority_count / majority_count < 0.7:
    df = augment(df)
    # Re-calcula las cantidades después de la ampliación
    label_counts = df['label'].value_counts()
    majority_count = label_counts.max()
    minority_count = label_counts.min()

# Punto de interrupción 2: Después de la Data Augmentation
print("Shape of the dataset after Data Augmentation:", df.shape)

# Aplicar Preprocesamiento
df['title'] = df['title'].apply(preprocess_text)

# Punto de interrupción 3: Después del Preprocesamiento
print("Shape of the dataset after Preprocessing:", df.shape)

# Aplicar Feature Engineering
df = feature_engineering(df)

# Punto de interrupción 4: Después del Feature Engineering
print("Shape of the dataset after Feature Engineering:", df.shape)

Y = df['label']
X = df.drop('label', axis=1)

# Punto de interrupción 5: Verificar las dimensiones de X_train e Y_train antes de la búsqueda en cuadrícula
print("Dimensiones de X:", X.shape)
print("Dimensiones de Y:", Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print("Dimensiones de X_train:", X_train.shape)
print("Dimensiones de Y_train:", Y_train.shape)

# Create a transformer that applies TfidfVectorizer to the 'title' column and StandardScaler to the numeric features.
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english')), 'title'),
        ('num', StandardScaler(), [col for col in X.columns if col != 'title'])
    ])

# Definir pipelines para cada modelo
pipelines = {
    'Logistic Regression': Pipeline([('preprocessor', preprocessor),
                                    ('clf', LogisticRegression(max_iter=1000))]),
    'LinearSVC': Pipeline([('preprocessor', preprocessor),
                            ('clf', LinearSVC(dual=False))])
}

# Definir parámetros para la búsqueda en cuadrícula
param_grid = {
    'preprocessor__text__max_df': [0.85, 0.9, 0.95],
    'preprocessor__text__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__C': [0.1, 1, 10]
}

# Optimizar hiperparámetros y evaluar cada modelo
best_estimators = {}
for name, pipeline in pipelines.items():
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, Y_train)
    best_estimators[name] = grid_search.best_estimator_

    y_pred = best_estimators[name].predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Accuracy del modelo {name} optimizado: {accuracy:.4f}")

    cv_accuracy = cross_val_score(best_estimators[name], X, Y, cv=5, scoring='accuracy').mean()
    print(f"Accuracy promedio con cross-validation para {name}: {cv_accuracy:.4f}\n")

# Modelo de ensemble
ensemble_model = VotingClassifier(estimators=[
    ('Logistic Regression', best_estimators['Logistic Regression']),
    ('LinearSVC', best_estimators['LinearSVC'])
], voting='hard')

ensemble_model.fit(X_train, Y_train)
ensemble_predictions = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(Y_test, ensemble_predictions)
print("\nAccuracy del modelo de ensemble:", ensemble_accuracy)

ensemble_cv_accuracy = cross_val_score(ensemble_model, X, Y, cv=5, scoring='accuracy').mean()
print(f"Accuracy promedio con cross-validation para el modelo de ensemble: {ensemble_cv_accuracy:.4f}\n")
