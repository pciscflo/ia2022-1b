#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Librerias
from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf


# In[2]:


# Fija las semillas aleatorias para la reproducibilidad
numpy.random.seed(7)
# carga los datos
dataset = numpy.loadtxt("data-diabetes.csv", delimiter=",")
# dividido en variables de entrada (X) y salida (Y)
X = dataset[:,0:8]
Y = dataset[:,8]


# In[8]:


# crea el modelo
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# In[4]:


#tf.keras.utils.plot_model(model, to_file="simple.png",show_shapes=True)


# In[6]:


# Compila el modelo, binary_crossentropy es la función de pérdida, entre el resultado predicho
# y el real.
# optimizador: Cambiando los pesos de las neuronas al final de la red neuronal utilizando el
# gradiente descendente para minimizar la función de pérdida. Algoritmo de descenso de gradiente
# estocástico “adam ”.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Entrena el modelo
model.fit(X, Y, epochs=150, batch_size=10) #para NO ver resultados del training poner verbose=0


# In[7]:


# calcula las predicciones
predictions = model.predict(X)
# redondeamos las predicciones
rounded = [round(x[0]) for x in predictions]
print(rounded)

