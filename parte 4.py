import numpy as np
import DnnLib
import json

def to_one_hot(labels):
    one_hot_labels = []

    for label in labels:
        out = [0] * 10
        out[label] = 1
        one_hot_labels.append(out)

    return np.array(one_hot_labels)

def load_data(train_path='mnist_train.npz', test_path='mnist_test.npz', scale=255, val_per=0.1):
    train_data = np.load(train_path)
    test_data = np.load(test_path)

    images = train_data['images'].reshape(train_data['images'].shape[0], -1) / scale
    labels = train_data['labels']

    test_images = test_data['images'].reshape(test_data['images'].shape[0], -1) / scale
    test_labels = test_data['labels']

    # -----------------------------
    N = images.shape[0]
    val_size = int(N * val_per)

    indices = np.arange(N)
    np.random.shuffle(indices)
    # -----------------------------

    val_indices, train_indices = indices[:val_size], indices[val_size:]

    val_images, val_labels = np.array(images[val_indices]), np.array(labels[val_indices])
    train_images, train_labels = np.array(images[train_indices]), np.array(labels[train_indices])

    train_oh = to_one_hot(train_labels)
    val_oh = to_one_hot(val_labels)
    test_oh = to_one_hot(test_labels)
        
    return (train_images, train_labels, train_oh), (val_images, val_labels, val_oh), (test_images, test_labels, test_oh)

def build_model():
    layers = [
        DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU),
        DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    ]

    optimizer = DnnLib.Adam(learning_rate=0.001)
    return layers, optimizer

def forward(layers, batch):
    out = batch
    for layer in layers:
        out = layer.forward(out)
    return out

def backward(layers, grad):
    for layer in reversed(layers):
        grad = layer.backward(grad)
        
def update_params(optimizer, layers):
    for layer in reversed(layers):
        optimizer.update(layer)
        
def get_data_batch(n_batch, batch_size, n_samples, train_data):
    start_idx = n_batch
    end_idx = min(n_batch + batch_size, n_samples)

    batch = train_data[0][start_idx:end_idx]
    batch_labels = train_data[1][start_idx:end_idx]
    batch_oh = train_data[2][start_idx:end_idx]
    return batch, batch_labels, batch_oh

def get_accuracy(output, one_hot_labels):
    predictions = np.argmax(output, axis=1)
    correct = np.argmax(one_hot_labels, axis=1)
    accuracy = (predictions == correct).mean()
    return accuracy

def get_epoch_metrics(epoch_loss, epoch_acc):
    avg_loss = np.array(epoch_loss).mean()
    avg_acc = np.array(epoch_acc).mean()
    return { "avg_loss": avg_loss, "avg_acc": avg_acc }

def model_evaluation(layers, data):
    # Calculo Forward
    output = forward(layers, data[0])
    
    # Calculo de Perdida
    loss = DnnLib.cross_entropy(output, data[2])
    
    # Calculo de Precision
    accuracy = get_accuracy(output, data[2])
    return output, loss, accuracy

def save_model(layers, input_space, preprocess, model_acc, filename="mnist_model_"):

    model = {
        "input_space": input_space,
        "preprocess": preprocess,
        "layers": [],
    }

    for layer in layers:
        layer_data = {
            "type" : "dense",
            "units" : len(layer.bias),
            "activation" : str(layer.activation_type).split(".")[1].lower(),
            "W" : layer.weights.T.tolist(),
            "b" : layer.bias.tolist(),
        }

        model['layers'].append(layer_data)

    filename = f"{filename}{model_acc:.3f}.json"

    with open(filename, "w") as f:
        json.dump(model, f)

    print(f'Model saved as {filename}')
    
def load_model(filename, test_data):
    with open(filename, 'r') as f:
        model_loaded = json.load(f)

    loaded_layers = model_loaded['layers']
    scale = model_loaded['preprocess']['scale']

    layers = []

    for layer in loaded_layers:
        activation = getattr(DnnLib.ActivationType, layer['activation'].upper())

        W = np.array(layer['W'])
        b = np.array(layer['b'])

        inputs = W.shape[0]
        neurons = W.shape[1]

        new_layer = DnnLib.DenseLayer(inputs, neurons, activation)

        new_layer.weights = W.T
        new_layer.bias = b

        layers.append(new_layer)

    _, _, test_acc = model_evaluation(layers, test_data)
    print('Model loaded correctly')
    print(f"Model Test | Accuracy: {(test_acc * 100):.2f}%")

    return layers

def train_model(layers, n_samples, train_data, val_data, epochs=15, batch_size=216):
    print("Starting model training...\n")
    print(f"With Epochs: {epochs} Batch size: {batch_size} \n")
    print("Epochs results: \n")
    
    for epoch in range(1, epochs + 1):
        epoch_loss, epoch_acc = [], []
        
        for n_batch in range(0, n_samples, batch_size):
            
            # Calculando el batch
            batch, batch_labels, batch_oh = get_data_batch(n_batch, batch_size, n_samples, train_data)

            # Calculo Forward
            output = forward(layers, batch)

            # Calculo de Perdida 
            loss = DnnLib.cross_entropy(output, batch_oh)
            
            # Calculo de Precision
            accuracy = get_accuracy(output, batch_oh)
            
            epoch_loss.append(loss)
            epoch_acc.append(accuracy)
            
            # Calculo de gradiente SCCE
            scce_gradient = DnnLib.cross_entropy_gradient(output, batch_oh)

            # Calculo Backward
            gradient = backward(layers, scce_gradient)

            # Actualizacion de parametros
            update_params(optimizer, layers)

        e_m = get_epoch_metrics(epoch_loss, epoch_acc)
        print(f"Epoch (Training) # {epoch} | Loss : {e_m['avg_loss']:.2f} | Accuracy: {(e_m['avg_acc'] * 100):.2f}%")

        _, val_loss, val_acc = model_evaluation(layers, val_data)
        print(f"Epoch (Val)      # {epoch} | Loss : {val_loss:.2f} | Accuracy: {(val_acc * 100):.2f}%")


train_data, val_data, test_data = load_data()

layers, optimizer = build_model()

train_model(layers, train_data[0].shape[0], train_data, val_data)

_, _, test_acc = model_evaluation(layers, test_data)

print(f"Model Test | Accuracy: {(test_acc * 100):.2f}%")

save_model(layers, [28, 28], {"scale": 255.0}, test_acc)