# Declaración de librerías
import json
import numpy
import gradio as gr

# Método para hacer la gráfica de cada modelo
# época a época para ver su evolución
import matplotlib.pyplot as plt


# El árbol conversacional debe de estar en un estado específico
# iniciar en el nivel contextual 1, e ir avanzando conforme a
# las decisiones que toma el usuario, como si fuera una máquina
# de estados
def instancer(inp, model, tags):
    inp = inp.lower().replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o")
    inp = inp.replace("ú", "u").replace("¿", "").replace("?", "")
    txt = [inp]
    seq = tokenizer_NI.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=maxlen)
    results = model.predict(padded)
    results_index = numpy.argmax(results)
    tag = list(tags.keys())[results_index]
    maxscore = numpy.max(results)
    return tag, maxscore


# Función del Nivel contextual 1:
def Activar_NI():
    # Pregunta inicial cuando se ingresa en el nivel contextual NI
    print("\nChatBot: ¿En qué puedo ayudarte?\n")

    while True:
        inp = input("     Tú: ")
        tag, maxscore = instancer(inp, model_NI, NI)

        if maxscore > 0.8 or inp == 'salir':
            break
        else:
            print("\nChatBot: Lo siento, pero no entendí tu petición, ¿Podrías decirlo de otra forma?\n")

    if inp == 'salir':
        print("\nChatBot: Hasta luego, fue un gusto hablar contigo\n")
        return 'salir'

    if tag == 'Reservar_Habitacion':
        if inp.count('Tipo_Habitacion') > 0:
            return 'NIIA1'
        elif inp.count('Fecha_entrada') > 0:
            return 'NIIA2'
        elif inp.count('Fecha_salida') > 0:
            return 'NIIA3'
        elif inp.count("Num_huespedes") > 0:
            return 'NIIA4'
        else:
            return 'NIIA'


# Función del Nivel contextual IIA:
def Activar_NIIA():
    # Pregunta inicial cuando se ingresa en el nivel contextual NIIA
    print("\nChatBot: Perfecto Vamos a reservar una habitación\n")

    # return 'NIIA1' ### De momento añado esta linea para quitar redundancia (perguntar dudas)

    global Tipo_Cuarto
    global Fecha_entrada
    global Fecha_salida
    global Num_huespedes

    while True:
        inp = input("     Tú: ")
        tag, maxscore = instancer(inp, model_NIIA, NIIA)

        if maxscore > 0.5 or inp == 'salir' or inp == 'volver':
            break
        else:
            print("\nChatBot: Lo siento, pero no entendí. Puedes decirme \"Reservar Habitación\" por ejemplo.\n")

    if inp == 'volver':
        return 'NI'

    if inp == 'salir':
        print("\nChatBot: Hasta luego, fue un gusto hablar contigo\n")
        return 'salir'

    if tag == 'Tipo_Habitacion' or Tipo_Cuarto == "":
        return 'NIIA1'

    if tag == 'Fecha_entrada':
        return 'NIIA2'

    if tag == 'Fecha_salida':
        return 'NIIA3'

    if tag == 'Num_huespedes':
        return 'NIIA4'


# Función del Nivel contextual IIA1:
def Activar_NIIA1():
    # Pregunta inicial cuando se ingresa en el nivel contextual NIIA1

    global Tipo_Cuarto

    print("\nChatBot: Excelente! Reservemos una habitación. ¿Qué tipo de habitación te gustaría reservar?\n")

    while True:
        inp = input("     Tú: ").lower()

        if inp == 'volver':
            return 'NI'
        elif inp == 'salir':
            print("\nChatBot: Hasta luego, fue un gusto hablar contigo\n")
            return 'salir'
        elif inp.count('estándar') > 0:
            print("\nChatBot: Excelente, reservaremos una habitación estándar.\n")
            Tipo_Cuarto = "Estándar"
            return 'NIIA2'
        elif inp.count('suite') > 0:
            print("\nChatBot: Excelente, reservaremos una habitación suite.\n")
            Tipo_Cuarto = "Suite"
            return 'NIIA2'
        elif inp.count('doble') > 0:
            print("\nChatBot: Excelente, reservaremos una habitación doble.\n")
            Tipo_Cuarto = "Doble"
            return 'NIIA2'
        elif inp.count('familiar') > 0:
            print("\nChatBot: Excelente, reservaremos una habitación familiar.\n")
            Tipo_Cuarto = "Familiar"
            return 'NIIA2'
        elif inp.count('deluxe') > 0:
            print("\nChatBot: Excelente, reservaremos una habitación deluxe.\n")
            Tipo_Cuarto = "Deluxe"
            return 'NIIA2'
        elif inp.count('dos camas') > 0:
            print("\nChatBot: Excelente, reservaremos una habitación doble.\n")
            Tipo_Cuarto = "Doble"
            return 'NIIA2'
        elif inp.count('ofrecen') > 0:
            print(
                "\nChatBot: Ofrecemos los siguientes tipos de habitaciones: estándar, doble, familiar, suite, y deluxe.\n")
            return 'NIIA2'
        elif inp.count('económica') > 0:
            print("\nChatBot: Excelente, reservaremos una habitación estándar.\n")
            Tipo_Cuarto = "Estándar"
            return 'NIIA2'
        else:
            print("\nChatBot: ¿Podrías repetirme qué tipo de habitación te gustaría reservar?\n")


# Función del Nivel contextual IIA2:
import locale
import platform
from datetime import datetime

# Configurar el locale en español
if platform.system() == 'Linux':
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
elif platform.system() == 'Windows':
    locale.setlocale(locale.LC_TIME, 'Spanish_Spain.1252')


def Activar_NIIA2():
    # Pregunta inicial cuando se ingresa en el nivel contextual NIIA2

    global Fecha_entrada

    print("\nChatBot: Muy bien! ¿Para cuándo agendamos su entrada? (dd/mm/aaaa)\n")

    while True:
        inp = input("     Tú: ").lower()

        if inp == 'volver':
            return 'NI'
        elif inp == 'salir':
            print("\nChatBot: Hasta luego, fue un gusto hablar contigo\n")
            return 'salir'
        else:
            # Intenta convertir la entrada en una fecha
            try:
                # Asume que la entrada está en formato dd/mm/aaaa
                fecha_entrada = datetime.strptime(inp, '%d/%m/%Y')
                # Comprueba que la fecha no sea pasada
                if fecha_entrada.date() < datetime.now().date():
                    print("\nChatBot: No puedes agendar una fecha en el pasado. Por favor, elige una fecha futura.\n")
                else:
                    fecha_formateada = fecha_entrada.strftime('%d de %B del %Y')
                    print("\nChatBot: Muy bien, agendaré la entrada para el {}.\n".format(fecha_formateada))
                    Fecha_entrada = fecha_entrada
                    return 'NIIA3'
            except ValueError:
                # Si hay un error en la conversión, indica que el formato no es correcto
                print(
                    "\nChatBot: Parece que la fecha no está en el formato correcto. Por favor, ingresa la fecha en formato dd/mm/aaaa.\n")


# Función del Nivel contextual IIA3:
def Activar_NIIA3():
    # Pregunta inicial cuando se ingresa en el nivel contextual NIIA3

    global Fecha_salida
    global Fecha_entrada

    print("\nChatBot: Muy bien! ¿Para cuándo agendamos su salida? (dd/mm/aaaa)\n")

    while True:
        inp = input("     Tú: ").lower()

        if inp == 'volver':
            return 'NI'
        elif inp == 'salir':
            print("\nChatBot: Hasta luego, fue un gusto hablar contigo\n")
            return 'salir'
        else:
            # Intenta convertir la entrada en una fecha
            try:
                # Asume que la entrada está en formato dd/mm/aaaa
                fecha_salida = datetime.strptime(inp, '%d/%m/%Y')
                # Comprueba que la fecha no sea pasada y que sea después de la fecha de entrada
                if fecha_salida.date() < datetime.now().date():
                    print(
                        "\nChatBot: No puedes agendar una fecha de salida en el pasado. Por favor, elige una fecha futura.\n")
                elif fecha_salida.date() <= Fecha_entrada.date():
                    print(
                        "\nChatBot: La fecha de salida debe ser posterior a la fecha de entrada. Por favor, elige una fecha adecuada.\n")
                else:
                    fecha_formateada = fecha_salida.strftime('%d de %B del %Y')
                    print("\nChatBot: Muy bien, agendaré la salida para el {}.\n".format(fecha_formateada))
                    Fecha_salida = fecha_salida
                    return 'NIIA4'
            except ValueError:
                # Si hay un error en la conversión, indica que el formato no es correcto
                print(
                    "\nChatBot: Parece que la fecha no está en el formato correcto. Por favor, ingresa la fecha en formato dd/mm/aaaa.\n")


# Función del Nivel contextual IIA4:
def Activar_NIIA4():
    # Pregunta inicial cuando se ingresa en el nivel contextual NIIA4

    global Num_huespedes

    print("\nChatBot: Súper! ¿Para cuántas personas es la reservación?\n")

    while True:
        inp = input("     Tú: ").lower()

        if inp == 'volver':
            return 'NI'
        elif inp == 'salir':
            print("\nChatBot: Hasta luego, fue un gusto hablar contigo\n")
            return 'salir'
        elif inp.count('yo') > 0:
            print("\nChatBot: Muy bien, agendaré para una sola persona.\n")
            Num_huespedes = 1
            print("La reservación final es de ", Num_huespedes, " personas con el tipo de cuarto ", Tipo_Cuarto,
                  " con fecha de entrada ", Fecha_entrada, " y fecha de salida ", Fecha_salida,
                  ". \n\n Si desdeas algo más no dudes en preguntar.")
            return 'NI'
        elif inp.count('solo') > 0:
            print("\nChatBot: Muy bien, agendaré para una sola persona.\n")
            Num_huespedes = 1
            print("La reservación final es de ", Num_huespedes, " personas con el tipo de cuarto ", Tipo_Cuarto,
                  " con fecha de entrada ", Fecha_entrada, " y fecha de salida ", Fecha_salida,
                  ". \n\n Si desdeas algo más no dudes en preguntar.")
            return 'NI'
        elif inp.count('2') > 0:
            print("\nChatBot: Muy bien, agendaré para dos personas.\n")
            Num_huespedes = 2
            print("La reservación final es de ", Num_huespedes, " personas con el tipo de cuarto ", Tipo_Cuarto,
                  " con fecha de entrada ", Fecha_entrada, " y fecha de salida ", Fecha_salida,
                  ". \n\n Si desdeas algo más no dudes en preguntar.")
            return 'NI'
        elif inp.count('dos') > 0:
            print("\nChatBot: Muy bien, agendaré para dos personas.\n")
            Num_huespedes = 2
            print("La reservación final es de ", Num_huespedes, " personas con el tipo de cuarto ", Tipo_Cuarto,
                  " con fecha de entrada ", Fecha_entrada, " y fecha de salida ", Fecha_salida,
                  ". \n\n Si desdeas algo más no dudes en preguntar.")
            return 'NI'
        elif inp.count('3') > 0:
            print("\nChatBot: Muy bien, agendaré para tres personas.\n")
            Num_huespedes = 3
            print("La reservación final es de ", Num_huespedes, " personas con el tipo de cuarto ", Tipo_Cuarto,
                  " con fecha de entrada ", Fecha_entrada, " y fecha de salida ", Fecha_salida,
                  ". \n\n Si desdeas algo más no dudes en preguntar.")
            return 'NI'
        elif inp.count('tres') > 0:
            print("\nChatBot: Muy bien, agendaré para tres personas.\n")
            Num_huespedes = 3
            print("La reservación final es de ", Num_huespedes, " personas con el tipo de cuarto ", Tipo_Cuarto,
                  " con fecha de entrada ", Fecha_entrada, " y fecha de salida ", Fecha_salida,
                  ". \n\n Si desdeas algo más no dudes en preguntar.")
            return 'NI'
        elif inp.count('4') > 0:
            print("\nChatBot: Muy bien, agendaré para cuatro personas.\n")
            Num_huespedes = 4
            print("La reservación final es de ", Num_huespedes, " personas con el tipo de cuarto ", Tipo_Cuarto,
                  " con fecha de entrada ", Fecha_entrada, " y fecha de salida ", Fecha_salida,
                  ". \n\n Si desdeas algo más no dudes en preguntar.")
            return 'salir'
        elif inp.count('cuatro') > 0:
            print("\nChatBot: Muy bien, agendaré para cuatro personas.\n")
            Num_huespedes = 4
            print("La reservación final es de ", Num_huespedes, " personas con el tipo de cuarto ", Tipo_Cuarto,
                  " con fecha de entrada ", Fecha_entrada, " y fecha de salida ", Fecha_salida,
                  ". \n\n Si desdeas algo más no dudes en preguntar.")
            return 'NI'


        else:
            print("\nChatBot: ¿Podrías repetirme para cuántas personas es la reservación?\n")



def Grafica_Modelo(history):
    # Parámetros de ploteo para la gráfica
    plt.figure(figsize=(12, 5))
    plt.ylim(-0.1, 1.1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Acc', 'Loss'])
    plt.show()




def Entrenar_Modelos(X_train, Y, model, labels):
    # Declaramos librería para convertir la salida en un vector
    # de X elementos con activación en la columna correspondiente
    # a su categoría
    train_labels = to_categorical(Y, num_classes=len(labels))
    # print('Matriz de salidas')
    # print(train_labels)

    # Ajuste de los datos de entrenamiento al modelo creado
    history = model.fit(X_train, train_labels, epochs=30, batch_size=1, verbose=1)

    # Cálculo de los procentajes de Eficiencia y pérdida
    score = model.evaluate(X_train, train_labels, verbose=1)
    # print("\nTest Loss:", score[0])
    # print("Test Accuracy:", score[1])
    return history


# Definición del método para tener la arquitectura de los modelos para cada nivel contextual
def Definir_Modelos(vocab_size, embedding_matrix, X_train, labels):
    # Declaración de las capas del modelo LSTM
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=X_train.shape[1],
                                trainable=False)
    model.add(embedding_layer)
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(len(labels), activation='softmax'))

    # Compilación del modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    print("\nPalabras en el vocabulario:")
    print(vocab_size)
    return model



# Asignamos los embeddings correspondientes a cada matriz
# con la que se entrenarán los modelos por medio de un método
def Asignar_Embeddings(tokenizer, vocab_size):
    # Generamos la matriz de embeddings (Con 300 Características)
    embedding_matrix = zeros((vocab_size, 300))
    for word, index in tokenizer.word_index.items():
        # Extraemos el vector de embedding para cada palabra
        embedding_vector = embeddings_dictionary.get(word)
        # Si la palbra si existía en el vocabulario
        # agregamos su vector de embeddings en la matriz
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix




# Para cada enunciado quitamos las StopWords
# También quitamos los acentos y filtramos signos de puntuación
# lo hacemos mediante un metodo
def Quitar_Stopwords(Textos):
    X = list()
    for sen in Textos:
        sentence = sen
        # Filtrado de stopword
        for stopword in stop_words:
            sentence = sentence.replace(" " + stopword + " ", " ")
        sentence = sentence.replace("á", "a")
        sentence = sentence.replace("é", "e")
        sentence = sentence.replace("í", "i")
        sentence = sentence.replace("ó", "o")
        sentence = sentence.replace("ú", "u")

        # Remover espacios múltiples
        sentence = re.sub(r'\s+', ' ', sentence)
        # Convertir todo a minúsculas
        sentence = sentence.lower()
        # Filtrado de signos de puntuación
        tokenizer = RegexpTokenizer(r'\w+')
        # Tokenización del resultado
        result = tokenizer.tokenize(sentence)
        # Agregar al arreglo los textos "destokenizados" (Como texto nuevamente)
        X.append(TreebankWordDetokenizer().detokenize(result))
    return X




def chatbot_interface(input_text):
    global Nivel, Num_huespedes, Tipo_Cuarto, Fecha_entrada, Fecha_salida
    if Nivel == 'NI':
        print("\nChatBot: Hola, soy el ChatBot, comienza a Hablar conmigo")

    Nivel = maquina_estados[Nivel](input_text)

    if Nivel == 'salir':
        return ("La reservación final es de {} personas con el tipo de cuarto {} con fecha de entrada {} y fecha de salida {}".format(Num_huespedes, Tipo_Cuarto, Fecha_entrada, Fecha_salida))
    else:
        return Nivel  # O alguna respuesta generada por tu chatbot


# Lectura de formatos .json para entrenar cada modelo y asignación
# de información correspondiente
with open('Intenciones_NivelI.json', encoding='utf-8') as file:
    data_NivelI = json.load(file)

with open('Intenciones_NivelIIA.json', encoding='utf-8') as file:
    data_NivelIIA = json.load(file)

with open('Intenciones_NivelIIB.json', encoding='utf-8') as file:
    data_NivelIIB = json.load(file)

# Creación de diccionarios con los nombres de las clases y textos
# presentes en cada uno de los archivos
NI = dict()
NIIA = dict()
NIIB = dict()

for info in data_NivelI['intents']:
    NI.setdefault(info['tag'], info['patterns'])

for info in data_NivelIIA['intents']:
    NIIA.setdefault(info['tag'], info['patterns'])

for info in data_NivelIIB['intents']:
    NIIB.setdefault(info['tag'], info['patterns'])

# print(NI)
# print(NIIA)
# print(NIIB)


# Generamos los vectores de respuestas para cada nivel contextual
# (Cada clase tiene una salida numérica asociada)
# (Las salidas empiezan en 0 para cada clase inicial de cada modelo)
Y_NI = list()
Y_NIIA = list()
Y_NIIB = list()

for clase, lista_textos in NI.items():
    for text in lista_textos:
        Y_NI.append(list(NI.keys()).index(clase))

for clase, lista_textos in NIIA.items():
    for text in lista_textos:
        Y_NIIA.append(list(NIIA.keys()).index(clase))

for clase, lista_textos in NIIB.items():
    for text in lista_textos:
        Y_NIIB.append(list(NIIB.keys()).index(clase))

# print("Vector de salidas Y para N1:")
# print(Y_NI)
# print("Vector de salidas Y para N2A:")
# print(Y_NIIA)
# print("Vector de salidas Y para N2B:")
# print(Y_NIIB)

# Importamos librerías para el filtrado de StopWords y tokenicación
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import re

stop_words = set(stopwords.words('spanish'))



# Obtenemos el vector de entradas (Textos limpios de StopWords)
# para cada uno de los modelos que vamos a generar
Textos_NI = list()
for Lista in NI.values():
    for Texto in Lista:
        Textos_NI.append(Texto)

X_NI = Quitar_Stopwords(Textos_NI)

Textos_NIIA = list()
for Lista in NIIA.values():
    for Texto in Lista:
        Textos_NIIA.append(Texto)

X_NIIA = Quitar_Stopwords(Textos_NIIA)

Textos_NIIB = list()
for Lista in NIIB.values():
    for Texto in Lista:
        Textos_NIIB.append(Texto)

X_NIIB = Quitar_Stopwords(Textos_NIIB)

# # Imprimimos la lista de los enunciados que resultan
# print(X_NI)
# print(X_NIIA)
# print(X_NIIB)

# Importamos la librería para generar la matriz de entrada de textos
# (El pad_sequence)
from keras_preprocessing.sequence import pad_sequences

# Cantidad de palabras máximas por ejemplo
# (Las más utilizadas)
maxlen = 5

# Preparamos la capa de embeddingsn(Predefinimos una cantidad de
# 5000 palabras consideradas como tokens
tokenizer_NI = Tokenizer(num_words=5000)
tokenizer_NIIA = Tokenizer(num_words=5000)
tokenizer_NIIB = Tokenizer(num_words=5000)

# Transforma cada texto en una secuencia de valores enteros para cada modelo que haremos
tokenizer_NI.fit_on_texts(X_NI)
X_NI_Tok = tokenizer_NI.texts_to_sequences(X_NI)
tokenizer_NIIA.fit_on_texts(X_NIIA)
X_NIIA_Tok = tokenizer_NIIA.texts_to_sequences(X_NIIA)
tokenizer_NIIB.fit_on_texts(X_NIIB)
X_NIIB_Tok = tokenizer_NIIB.texts_to_sequences(X_NIIB)

# Especificamos la matriz (Con padding hasta maxlen)
X_NI_train = pad_sequences(X_NI_Tok, padding='post', maxlen=maxlen)
X_NIIA_train = pad_sequences(X_NIIA_Tok, padding='post', maxlen=maxlen)
X_NIIB_train = pad_sequences(X_NIIB_Tok, padding='post', maxlen=maxlen)
#
# print("Matriz de entrada para NI:")
# print(X_NI_train)
#
# print("Matriz de entrada para NIIA:")
# print(X_NIIA_train)
#
# print("Matriz de entrada para NIIB:")
# print(X_NIIB_train)

# Declaración de librerías para manejo de arreglos (Numpy)
from numpy import asarray
from numpy import zeros

# Lectura del archivo de embeddings
embeddings_dictionary = dict()
Embeddings_file = open('Word2Vect_Spanish.txt', encoding="utf8")

# Extraemos las características del archivo de embeddings
# y las agregamos a un diccionario (Cada elemento es un vextor)
for linea in Embeddings_file:
    caracts = linea.split()
    palabra = caracts[0]
    vector = asarray(caracts[1:], dtype='float32')
    embeddings_dictionary[palabra] = vector
Embeddings_file.close()



# Obtenemos las matrices de Embeddings para cada modelo
# Y también el tamaño del vocabulario para cada uno
vocab_size_NI = len(tokenizer_NI.word_index) + 1
embedding_matrix_NI = Asignar_Embeddings(tokenizer_NI, vocab_size_NI)

vocab_size_NIIA = len(tokenizer_NIIA.word_index) + 1
embedding_matrix_NIIA = Asignar_Embeddings(tokenizer_NIIA, vocab_size_NIIA)

vocab_size_NIIB = len(tokenizer_NIIB.word_index) + 1
embedding_matrix_NIIB = Asignar_Embeddings(tokenizer_NIIB, vocab_size_NIIB)

# Declaración de modelo Secuencial que usaremos para todos los casos
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers import Embedding



# Generamos la arquitectura para el modelo de NI
model_NI = Definir_Modelos(vocab_size_NI, embedding_matrix_NI, X_NI_train, NI.keys())

# Generamos la arquitectura para el modelo de NIIA
model_NIIA = Definir_Modelos(vocab_size_NIIA, embedding_matrix_NIIA, X_NIIA_train, NIIA.keys())

# Generamos la arquitectura para el modelo de NIIB
model_NIIB = Definir_Modelos(vocab_size_NIIB, embedding_matrix_NIIB, X_NIIB_train, NIIB.keys())

# Declaramos el método para entrenar cada modelo
from keras.utils.np_utils import to_categorical


# Entrenamos el modelo del nivel NI y obtenemos el historial de las épocas para realizar su gráfica
history_NI = Entrenar_Modelos(X_NI_train, Y_NI, model_NI, NI.keys())

# Entrenamos el modelo del nivel NIIA y obtenemos el historial de las épocas para realizar su gráfica
history_NIIA = Entrenar_Modelos(X_NIIA_train, Y_NIIA, model_NIIA, NIIA.keys())

# Entrenamos el modelo del nivel NIIB y obtenemos el historial de las épocas para realizar su gráfica
history_NIIB = Entrenar_Modelos(X_NIIB_train, Y_NIIB, model_NIIB, NIIB.keys())



# Graficar el modelo NI
Grafica_Modelo(history_NI)

# Graficar el modelo NIIA
Grafica_Modelo(history_NIIA)

# Graficar el modelo NIIB
Grafica_Modelo(history_NIIB)



# Implementación de casos correspondientes para cada nivel del ChatBot
# Nivel contextual inicial por defecto, el primero

maquina_estados = {'NI': Activar_NI,
                   'NIIA': Activar_NIIA,
                   'NIIA1': Activar_NIIA1,
                   'NIIA2': Activar_NIIA2,
                   'NIIA3': Activar_NIIA3,
                   'NIIA4': Activar_NIIA4
                   }

Tipo_Cuarto = ""
Fecha_entrada = ""
Fecha_salida = ""
Num_huespedes = 0

def chat1():
    Nivel = 'NI'
    # Pregunta inicial, solo cuando se inicia el ChatBot
    print("\nChatBot: Hola, soy el ChatBot, comienza a Hablar conmigo")

    while True:
        Nivel = maquina_estados[Nivel]()
        print(Nivel)
        if Nivel == 'salir':
            print("La reservación final es de ", Num_huespedes, " personas con el tipo de cuarto ", Tipo_Cuarto,
                  " con fecha de entrada ", Fecha_entrada, " y fecha de salida ", Fecha_salida)
            break


# Inicializa las variables globales
Nivel = 'NI'
Num_huespedes = 0
Tipo_Cuarto = ''
Fecha_entrada = ''
Fecha_salida = ''

# Suponiendo que tienes una función que maneja la lógica del chatbot y devuelve una respuesta
def manejar_mensaje(mensaje):
    global Nivel
    respuesta = ""
    Nivel = maquina_estados[Nivel](mensaje)
    if Nivel == 'salir':
        respuesta = f"La reservación final es de {Num_huespedes} personas con el tipo de cuarto {Tipo_Cuarto}, con fecha de entrada {Fecha_entrada} y fecha de salida {Fecha_salida}"
        Nivel = 'NI'
    else:
        respuesta = "..."
    return respuesta

# Esta es la función que Gradio llamará cuando el usuario envíe un mensaje
def chatbot_interface(mensaje):
    try:
        return manejar_mensaje(mensaje)
    except Exception as e:
        return str(e)  # Devuelve el mensaje de error para depuración

# Configura la interfaz de Gradio
iface = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Escribe aquí para hablar con el ChatBot..."),
    outputs="text",
    live=True
)

# Inicia la interfaz de Gradio
iface.launch()
