"""
Laboratorio 05 - Arquitecturas CNN y Analisis de Resultados
PARTE 1: Practica con Digitos Escritos a Mano (MNIST)
Curso: SIS070 - Inteligencia Artificial

Entrena una Red Neuronal Convolucional (CNN) para clasificar digitos del 0 al 9
del dataset MNIST, y analiza su arquitectura, sus parametros y su prediccion
sobre una muestra de prueba.
"""

import os

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("--- [PARTE 1] INICIO DE EJECUCION: MNIST ---")

# ----------------------------------------------------------------------
# 1. Carga y preprocesamiento del dataset MNIST
# 60 000 imagenes de entrenamiento + 10 000 de prueba, cada una de 28x28
# pixeles en escala de grises, ya etiquetadas con el digito que representan.
# ----------------------------------------------------------------------
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# ----------------------------------------------------------------------
# ACTIVIDAD ADICIONAL: Visualizacion de muestras del Dataset
# Muestra una cuadricula 3x3 con las primeras 9 imagenes de entrenamiento
# y su etiqueta real, para inspeccionar visualmente los datos crudos antes
# de alimentarlos a la CNN.
# ----------------------------------------------------------------------
print("\n--- Mostrando imagenes de ejemplo del dataset MNIST ---")
plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Etiqueta: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()

# Se guarda la figura en disco ademas de mostrarla: una ventana interactiva
# (plt.show) no puede incluirse directamente en el informe, asi que se
# persiste como imagen para usarla como evidencia en el README.
os.makedirs('img', exist_ok=True)
plt.savefig('img/visualizacion_mnist.png', dpi=150, bbox_inches='tight')
plt.show()  # Se abrira una ventana con la cuadricula de imagenes de entrenamiento

# ----------------------------------------------------------------------
# Normalizacion y dimension de canal
# Se escalan los pixeles de [0, 255] a [0.0, 1.0] (las redes convergen mejor
# con entradas pequenas) y se agrega una dimension de canal al final
# (28, 28) -> (28, 28, 1), porque Conv2D espera imagenes con canales
# explicitos (1 = escala de grises, 3 seria RGB).
# ----------------------------------------------------------------------
X_train = X_train.astype("float32") / 255.0
X_train = np.expand_dims(X_train, -1)

X_test = X_test.astype("float32") / 255.0
X_test = np.expand_dims(X_test, -1)

# ----------------------------------------------------------------------
# 2. Construccion de la Arquitectura CNN
# Dos bloques Convolucion + MaxPooling (extraen caracteristicas espaciales
# y reducen la resolucion progresivamente), seguidos de un Flatten que
# convierte el mapa de caracteristicas 2D en un vector, y 2 capas densas
# (una oculta y una de salida con Softmax para las 10 clases de digitos).
# ----------------------------------------------------------------------
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), name='Conv_1'),
    layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool_1'),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='Conv_2'),
    layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool_2'),
    layers.Flatten(name='Aplanado_Flatten'),
    layers.Dense(64, activation='relu', name='Capa_Densa_Oculta'),
    layers.Dense(10, activation='softmax', name='Salida_Softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------------------------------------------------------------
# 3. Entrenamiento (1 epoca de demostracion)
# Se entrena solo con las primeras 5000 muestras y 1 epoca, suficiente para
# demostrar el flujo completo (forward, calculo de perdida, backpropagation)
# sin esperar un entrenamiento largo.
# ----------------------------------------------------------------------
print("\n--- Entrenando el modelo MNIST (5000 muestras, 1 epoca) ---")
history = model.fit(X_train[:5000], y_train[:5000], epochs=1, batch_size=64)

# ----------------------------------------------------------------------
# 4. Analisis y Prediccion de una muestra de prueba
# Se toma la primera imagen del set de prueba (nunca vista en entrenamiento)
# y se le pide al modelo su vector de probabilidades (Softmax) sobre las
# 10 clases; la clase predicha es el indice con mayor probabilidad.
# ----------------------------------------------------------------------
print("\n--- Analizando Resultados de Prediccion (MNIST) ---")
imagen_prueba = X_test[0:1]
predicciones = model.predict(imagen_prueba)

print(f"Vector de probabilidades crudas (Softmax): \n{predicciones[0].round(4)}")
print(f"Clase predicha por el modelo: {np.argmax(predicciones)}")
print(f"Clase real de la imagen: {y_test[0]}")
