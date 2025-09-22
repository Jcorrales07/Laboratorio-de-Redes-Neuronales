# Red Neuronal MNIST + DnnLib + NumPy 

Pequeño proyecto modular para entrenar una red neuronal MLP sobre MNIST usando **DnnLib** (libreria creada por el ingeniero Ivan Deras) dentro de **Docker**. Incluye **dropout** y **regularización L1/L2**, guardado/carga del modelo en JSON y CLI con `argparse`.

---

## 🗂 Estructura

```
/app
├─ main.py               # CLI: entrena, evalúa en test y guarda el modelo
├─ neural_network.py     # definición del modelo, forward/backward, training loop
├─ utils.py              # carga de datos, one-hot, métricas, guardar/cargar JSON
├─ dataset               # dataset para el entrenamiento y prueba de modelos
    ├─ fashion_mnist/
        ├─  fashion_mnist_train.npz 
        └─  fashion_mnist_train.npz
    ├─ mnist/
        ├─  mnist_train.npz 
        └─  mnist_train.npz   
├─ modelos_entrenados    # carpeta para guardar modelos
    ├─ fashion_mnist/
    ├─ mnist/
```

---

## 🔧 Requisitos

* Docker instalado
* Imagen del curso:

  ```bash
  docker pull iderashn/dnn-q32025:latest
  ```
* Para ejecutar el notebook:
  ```bash
  docker run --rm -p 8888:8888 iderashn/dnn-q32025:latest
  ```
* Archivos `mnist_train.npz`, `mnist_test.npz`, `fashion_mnist_train.npz` y `fashion_mnist_test.npz` en la carpeta donde correrás el script.

---

## ▶️ Cómo correrlo dentro de Docker

### Opción A — Ejecutar directamente (recomendada)

```bash
  python3 main.py
```

**Windows (PowerShell):**

```powershell
docker run -it --rm `
  -v "${PWD}:/work" -w /work `
  iderashn/dnn-q32025:latest `
  python main.py
```
```
O puedes correrlo en la ventana `Exec` y correr `python main.py` y poner el path de los archivos .npz para empezar
```
![alt text](image.png)

---

## ⚙️ Arquitectura del modelo

* `Dense(784 → 128, ReLU)`
* `Dropout(p = --dropout)` **solo en entrenamiento**
* `Dense(128 → 10, Softmax)`
* Optimizador: `Adam(learning_rate)`
* Regularización por capa: `L2`

---

## 📦 CLI (argumentos)

```text
--train_path <str>         Ruta a mnist_train.npz (default: mnist_train.npz)
--test_path <str>          Ruta a mnist_test.npz (default: mnist_test.npz)
--scale <float>            Escala para normalizar imágenes (default: 255.0)
--val_per <float>          Proporción para validación (default: 0.1)

--epochs <int>             Épocas de entrenamiento (default: 15)
--batch_size <int>         Tamaño de batch (default: 216)
--learning_rate <float>    LR para Adam (default: 0.001)

--filename <str>           Prefijo/nombre para guardar el modelo (default: mnist_model_)
```

### Ejemplos

```bash
python3 main.py --epochs 15 --batch_size 216 --learning_rate 0.001
```

Cambiar escala y split:

```bash
python3 main.py --scale 255.0 --val_per 0.1
```

Guardar con nombre fijo:

```bash
python3 main.py --filename my_mnist.json
```

> Si `--filename` es `mnist_model_` o `fashion_mnist_model_`, el script añade el accuracy al nombre, ej. `mnist_model_0.932.json`.

---

## 🧪 Salida y archivos generados

* En consola: loss/accuracy por época en train/val y accuracy final en test.
* Archivo JSON con pesos y metadatos. Estructura típica:

  ```json
  {
    "input_space": [28, 28],
    "preprocess": { "scale": 255.0 },
    "layers": [
      {
        "type": "dense",
        "units": 128,
        "activation": "relu",
        "W": [[...]],  // pesos guardados como (inputs, neurons) == weights.T
        "b": [...]
      },
      {
        "type": "dense",
        "units": 10,
        "activation": "softmax",
        "W": [[...]],
        "b": [...]
      }
    ]
  }
  ```
* **Dropout no se guarda** (no tiene parámetros).

---

## 📥 Cargar y evaluar un modelo guardado (snippet opcional)

No hay subcomando `eval` en CLI, pero puedes usarlo así dentro de Python:

```python
import numpy as np
from utils import load_data, load_model
from neural_network import model_evaluation

# Carga datos
_, _, test_data = load_data('mnist_train.npz', 'mnist_test.npz', scale=255.0, val_per=0.1)

# Carga modelo desde JSON
layers = load_model('mnist_model_0.932.json')

# Evalúa en test
_, _, test_acc = model_evaluation(layers, test_data)
print(f"Test Accuracy: {test_acc*100:.2f}%")
```

---

## 🧩 Notas importantes

* **Tipos correctos**:

  * `--learning_rate` debe ser `float` (si lo pones como `int`, se vuelve `0` y la red no aprende).
  * `set_regularizer` **solo acepta posicionales**: `layer.set_regularizer(RegularizerType.L2, reg_lambda)`.
* **Evitar imports circulares**:

  * `utils.py` no importa `forward`. `model_evaluation` vive en `neural_network.py`.
* **Dropout**:

  * Se activa solo en entrenamiento (`forward(..., training=True)`), se desactiva en evaluación (`training=False`).
* **Loss de regularización**:

  * Se suma solo para **reporte** (los gradientes ya incluyen el término por `set_regularizer`).
* **Escalado**:

  * Imágenes normalizadas con `/ scale` (por defecto `255.0`). Asegúrate que `preprocess.scale` coincide con lo que guardas.

---

## 📈 Rendimiento esperado

Con `--epochs 15 --batch_size 216 --learning_rate 0.001 --reg_lambda 0.001`, deberías ver accuracies de validación/test **\~90%+** en MNIST (varía por semilla y entorno). Si te quedas cerca de 10%:

* Verifica que el LR no sea 0.
* Revisa que `np.argmax(..., axis=1)` se use en precisión.
* Confirma que el dataset `.npz` tiene forma esperada y que `--scale` sea `255.0`.
