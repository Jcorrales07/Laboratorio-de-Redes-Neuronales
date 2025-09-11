# PARTE 2
import numpy as np
data = np.load("mnist_train.npz")

images = data["images"]
labels = data["labels"]

import matplotlib.pyplot as plt

num_imgs = 9
cols = 5
rows = int(np.ceil(num_imgs / cols))

plt.figure(figsize=(8, 8))

for i in range(num_imgs):
    plt.subplot(rows, cols, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(f"Label: {labels[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()