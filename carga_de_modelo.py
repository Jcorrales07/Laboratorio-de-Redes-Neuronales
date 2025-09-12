# Parte 3, Cargar un modelo pre entrenado

import numpy as np
import json # De un archivo json viene el modelo
import DnnLib
import matplotlib as plt
import math
import random 

# Cargamos el archivo .json en donde esta todos los pesos del modelo
with open('./modelos_entrenados/mnist_mlp_pretty.json', 'r') as f:
    model = json.load(f)
    
# Hay que reconstruir las capas de la red
layers = []

for layer in model['layers']:
    if layer['type'] == "dense":
        activation = getattr(DnnLib.ActivationType, layer['activation'].upper())
        
        # Conseguimos los parametros
        W = np.array(layer['W'], dtype=np.float32)
        b = np.array(layer['b'], dtype=np.float32)
        
        in_features = W.shape[0]
        out_features = W.shape[1]
        
        # Creamos la capa, con la cantidad de entradas y neuronas (salidas)
        dense = DnnLib.DenseLayer(in_features, out_features, activation)
        
        # Aca asignamos los pesos y los sesgos de la capa
        # Transpuesto para que se hagan las operaciones por columna
        dense.weights = W.T
        dense.bias = b
        
        layers.append(dense)
        
        
# Definimos la funcion forward
def forward(x):
    """x shape must be (784, )"""
    out = x
    for layer in layers:
        out = layer.forward(out)
        return out

# Cargamos el dataset de prueba    
data = np.load("./dataset/mnist_test.npz")
scale = model['preprocess']['scale'] # Extraemos el scale del modelo
images_test = data['images'] / scale
labels_test = data['labels']


correct = 0
for i in range(len(images_test)):
    x = images_test[i].reshape(1, -1) # Aplanamos la imagen 28x28 a 784 pixeles
    y_pred = forward(x)
    if np.argmax(y_pred) == labels_test[i]:
        correct += 1
        
accuracy = correct / len(images_test)
print(f"Precision en dataset de test: {accuracy * 100:.2f}%")

num_imgs = 28
cols = 4
rows = int(np.ceil(num_imgs / cols))

plt.figure(figsize=(8, 8))

for i in range(num_imgs):
    random_idx = math.ceil(random.random() * num_imgs + 1)
    sample = images_test[random_idx].reshape(-1)
    pred = forward(sample)
    plt.subplot(rows, cols, i+1)
    plt.imshow(images_test[random_idx], cmap='gray')
    
    color = 'green' if labels_test[random_idx] == np.argmax(pred) else 'red'
    plt.title(f"Label: {labels_test[random_idx]} \nPred: {np.argmax(pred)}", color=color)
    plt.axis("off")

plt.tight_layout()
plt.show()