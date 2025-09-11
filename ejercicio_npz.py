# Ejercicio
# Cargar mnis_test.npz
# Mostrar las primeras 16 imagenes en una cuadricula 4x4
# Comparar las etiquetas reales (labels) con lo que ven en las imagenes

import numpy as np
import matplotlib as plt


data = np.load("./dataset/mnist_test.npz")
imgs = data['images']
lbls = data['labels']
print(imgs.shape, lbls.shape)

num_imgs = 16
cols = 4
rows = int(np.ceil(num_imgs / cols))

plt.figure(figsize=(8, 8))

for i in range(num_imgs):
    plt.subplot(rows, cols, i+1)
    plt.imshow(imgs[i], cmap='gray')
    plt.title(f"Label: {lbls[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()