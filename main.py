import argparse
from utils import load_data, save_model
from neural_network import build_model, train_model, model_evaluation

parser = argparse.ArgumentParser(
    description="Entrena una RN MNIST con Numpy DnnLib"
)

parser.add_argument("--train_path", type=str, default='mnist_train.npz', help="Ruta al dataset de entrenamiento")
parser.add_argument("--test_path", type=str, default='mnist_test.npz', help="Ruta al dataset de evaluación")
parser.add_argument("--scale", type=float, default=255.0, help="Escala para preprocesar los datos del dataset")
parser.add_argument("--val_per", type=float, default=0.1, help="Porcentaje para el dataset para la validacion de las epocas")

parser.add_argument("--epochs", type=int, default=15, help="Número de epocas para el entrenamiento")
parser.add_argument("--batch_size", type=int, default=216, help="Número de epocas para el entrenamiento")
parser.add_argument("--learning_rate", type=int, default=0.001, help="Numero sobre la tasa de aprendizaje de la Red Neuronal")

parser.add_argument("--filename", type=str, default="mnist_model_", help="Nombre del archivo para guardar el modelo")
args = parser.parse_args()

train_path: str = args.train_path
test_path: str = args.test_path
scale: int = args.scale
val_per: float =args.val_per

epochs: int = args.epochs
batch_size: int = args.batch_size
learning_rate: float = args.learning_rate

filename: str = args.filename

train_data, val_data, test_data = load_data(train_path=train_path, test_path=test_path, scale=scale, val_per=val_per)

layers, optimizer = build_model(learning_rate=learning_rate)

train_model(layers, train_data[0].shape[0], optimizer, train_data, val_data, epochs=epochs, batch_size=batch_size)

_, _, test_acc = model_evaluation(layers, test_data)

print(f"Model Test | Accuracy: {(test_acc * 100):.2f}%")

save_model(layers, [28, 28], {"scale": 255.0}, test_acc, filename=filename)