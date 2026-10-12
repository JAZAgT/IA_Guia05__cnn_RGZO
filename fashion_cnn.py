"""
Laboratorio 05 - Arquitecturas CNN y Analisis de Resultados
PARTE 2: Practica con Clasificacion de Prendas (Fashion-MNIST)
Curso: SIS070 - Inteligencia Artificial

Misma arquitectura base que mnist_cnn.py, aplicada ahora a Fashion-MNIST
(10 categorias de prendas de vestir), con la capa densa oculta ampliada
de 64 a 128 neuronas para evaluar su impacto en la cantidad de parametros.
"""

import os

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("--- [PARTE 2] INICIO DE EJECUCION: FASHION-MNIST ---")

# ----------------------------------------------------------------------
# 1. Carga y preprocesamiento de Fashion-MNIST
# Mismo formato que MNIST (60 000 + 10 000 imagenes de 28x28 en escala de
# grises), pero cada imagen es una prenda de vestir en vez de un digito.
# class_names traduce la etiqueta numerica (0-9) a un nombre legible.
# ----------------------------------------------------------------------
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

class_names = ['Camiseta/Top', 'Pantalon', 'Pulover', 'Vestido', 'Abrigo',
               'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Boton']

# ----------------------------------------------------------------------
# ACTIVIDAD ADICIONAL: Visualizacion de muestras del Dataset
# Igual que en MNIST, pero el titulo de cada imagen usa class_names en vez
# del numero crudo, para poder inspeccionar visualmente las prendas.
# ----------------------------------------------------------------------
print("\n--- Mostrando imagenes de ejemplo del dataset Fashion-MNIST ---")
plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"{class_names[y_train[i]]}")
    plt.axis('off')
plt.tight_layout()

# Se guarda la figura en disco ademas de mostrarla (ver nota en mnist_cnn.py).
os.makedirs('img', exist_ok=True)
plt.savefig('img/visualizacion_fashion.png', dpi=150, bbox_inches='tight')
plt.show()  # Se abrira una ventana con la cuadricula de prendas de vestir

# ----------------------------------------------------------------------
# Normalizacion y dimension de canal (identico a MNIST)
# ----------------------------------------------------------------------
X_train = X_train.astype("float32") / 255.0
X_train = np.expand_dims(X_train, -1)

X_test = X_test.astype("float32") / 255.0
X_test = np.expand_dims(X_test, -1)

# ----------------------------------------------------------------------
# 2. Construccion de la Arquitectura CNN (con capa densa ampliada)
# Misma base convolucional que MNIST, pero la capa densa oculta pasa de
# 64 a 128 neuronas: se compara el impacto de este cambio en el numero
# total de parametros en el README (Parte 2, pregunta 3).
# ----------------------------------------------------------------------
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), name='Conv_1'),
    layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool_1'),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='Conv_2'),
    layers.MaxPooling2D(pool_size=(2, 2), name='MaxPool_2'),
    layers.Flatten(name='Aplanado_Flatten'),
    layers.Dense(128, activation='relu', name='Capa_Densa_Oculta'),  # Capa ampliada a 128
    layers.Dense(10, activation='softmax', name='Salida_Softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------------------------------------------------------------
# 3. Entrenamiento (1 epoca de demostracion)
# ----------------------------------------------------------------------
print("\n--- Entrenando el modelo Fashion-MNIST (5000 muestras, 1 epoca) ---")
history = model.fit(X_train[:5000], y_train[:5000], epochs=1, batch_size=64)

# ----------------------------------------------------------------------
# 4. Analisis y Prediccion de una muestra de prueba
# ----------------------------------------------------------------------
print("\n--- Analizando Resultados de Prediccion (Fashion-MNIST) ---")
imagen_prueba = X_test[0:1]
predicciones = model.predict(imagen_prueba)
indice_predicho = np.argmax(predicciones)

print(f"Vector de probabilidades crudas (Softmax): \n{predicciones[0].round(4)}")
print(f"Clase predicha: {class_names[indice_predicho]} (indice {indice_predicho})")
print(f"Clase real: {class_names[y_test[0]]} (indice {y_test[0]})")
