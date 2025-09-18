import numpy as np
import DnnLib
import matplotlib.pyplot as plt

# Cargar los datos MNIST
data = np.load('mnist_train.npz')
X_train = data['images']
y_train = data['labels']

X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float64)
X_train = X_train / 255.0

# Convertir etiquetas a one-hot encoding para 10 clases (dígitos 0-9)
n_classes = 10
n_samples = y_train.shape[0]
y_train_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
y_train_onehot[np.arange(n_samples), y_train] = 1.0

layers = [
    DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU),    # Capa oculta
    DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)   # Capa de salida
]

optimizer = DnnLib.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

# Función para calcular precisión
def calculate_accuracy(predictions, true_labels):
    """Calcula la precisión del modelo"""
    predicted_classes = np.argmax(predictions, axis=1)
    return np.mean(predicted_classes == true_labels)

# Entrenamiento del modelo
print("\nIniciando entrenamiento...")
epochs = 100
batch_size = 512  # Mini-batch para mejor rendimiento

# Listas para almacenar métricas
train_losses = []
train_accuracies = []

n_samples = X_train.shape[0]
n_batches = (n_samples + batch_size - 1) // batch_size

for epoch in range(epochs):
    # Mezclar los datos en cada época
    indices = np.random.permutation(n_samples) 
    X_shuffled = X_train[indices]
    y_shuffled = y_train_onehot[indices]
    y_labels_shuffled = y_train[indices]
    
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    
    # Procesamiento por mini-batches
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        y_labels_batch = y_labels_shuffled[start_idx:end_idx]
        
        # Forward pass
        h1 = layers[0].forward(X_batch)  # Capa oculta (ReLU)
        output = layers[1].forward(h1)   # Capa de salida (Softmax)
        
        # Cálculo de la pérdida (Cross Entropy)
        loss = DnnLib.cross_entropy(output, y_batch)
        
        # Backward pass
        # Aca no se ocupa derivada de softmax, el ing dio una razon
        # El ing nos dijo que probaramos hacer esta funcion nosotros mismos y probar

        # Solo en el caso especial de que se usa SOFTMAX en la capa de salida junto con CCE se calcula de un solo el gradiente multiplicado de los 2
        # Es el unico caso en el que la regla de la cadena no funciona, por que sale un mega gradiente y de  esta manera se simplifica
        grad = DnnLib.cross_entropy_gradient(output, y_batch)
        grad = layers[1].backward(grad)  # Gradiente para capa de salida
        grad = layers[0].backward(grad)  # Gradiente para capa oculta
        
        # Actualizar parámetros
        optimizer.update(layers[1])
        optimizer.update(layers[0])
        
        # Acumular métricas
        epoch_loss += loss
        epoch_accuracy += calculate_accuracy(output, y_labels_batch)
    
    # Promediar métricas de la época
    avg_loss = epoch_loss / n_batches
    avg_accuracy = epoch_accuracy / n_batches
    
    train_losses.append(avg_loss)
    train_accuracies.append(avg_accuracy)
    
    # Mostrar progreso cada 10 épocas
    if (epoch + 1) % 10 == 0:
        print(f"Época {epoch + 1}/{epochs} - "
              f"Pérdida: {avg_loss:.4f} - "
              f"Precisión: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")

# Evaluación final del modelo
print("\nEvaluación final en todo el conjunto de entrenamiento...")

# Forward pass completo
h1_final = layers[0].forward(X_train)
output_final = layers[1].forward(h1_final)

# Métricas finales
final_loss = DnnLib.cross_entropy(output_final, y_train_onehot)
final_accuracy = calculate_accuracy(output_final, y_train)

print(f"\nResultados finales:")
print(f"Pérdida final: {final_loss:.4f}")
print(f"Precisión final: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

if final_accuracy >= 0.85:
    print("✅ ¡Objetivo alcanzado! Precisión superior al 85%")
else:
    print("❌ Precisión por debajo del objetivo del 85%")

# Visualización de las métricas de entrenamiento
plt.figure(figsize=(12, 4))

# Gráfico de pérdida
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, 'b-', linewidth=2)
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Cross Entropy Loss')
plt.grid(True, alpha=0.3)

# Gráfico de precisión
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, 'r-', linewidth=2)
plt.axhline(y=0.85, color='g', linestyle='--', label='Objetivo (85%)')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Análisis de predicciones por clase
print("\nAnálisis de predicciones por clase:")
predicted_classes = np.argmax(output_final, axis=1)

for class_idx in range(10):
    class_mask = y_train == class_idx
    class_accuracy = np.mean(predicted_classes[class_mask] == class_idx)
    class_count = np.sum(class_mask)
    print(f"Dígito {class_idx}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - "
          f"{class_count} muestras")

# Mostrar algunos ejemplos de predicciones
print("\nEjemplos de predicciones:")
sample_indices = np.random.choice(n_samples, 5, replace=False)

for idx in sample_indices:
    true_label = y_train[idx]
    predicted_label = predicted_classes[idx]
    confidence = output_final[idx, predicted_label]
    
    status = "✅" if true_label == predicted_label else "❌"
    print(f"{status} Muestra {idx}: Verdadero={true_label}, "
          f"Predicho={predicted_label}, Confianza={confidence:.4f}")

print(f"\nEntrenamiento completado exitosamente!")
print(f"Arquitectura final: 784 -> 128 (ReLU) -> 10 (Softmax)")
print(f"Optimizador: Adam")
print(f"Pérdida: Cross Entropy")
print(f"Precisión alcanzada: {final_accuracy*100:.2f}%")