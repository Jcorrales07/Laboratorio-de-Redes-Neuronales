import json
import numpy as np
import DnnLib

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



def save_model(layers, input_space, preprocess, model_acc, filename="mnist_model_"):

    model = {
        "input_space": input_space,
        "preprocess": preprocess,
        "layers": [],
    }

    for layer in layers:
        if hasattr(layer, "training"):
            continue
        
        layer_data = {
            "type" : "dense",
            "units" : len(layer.bias),
            "activation" : str(layer.activation_type).split(".")[1].lower(),
            "W" : layer.weights.T.tolist(),
            "b" : layer.bias.tolist(),
        }

        model['layers'].append(layer_data)

    if filename == 'mnist_model_' or filename == 'fashion_mnist_model_':
        filename = f"{filename}{model_acc:.3f}.json"
    else:
        filename = filename.split('.')[0]
        filename = f"{filename}.json"

    with open(filename, "w") as f:
        json.dump(model, f)

    print(f'Model saved as {filename}')
    
def load_model(filename, test_data):
    with open(filename, 'r') as f:
        model_loaded = json.load(f)

    loaded_layers = model_loaded['layers']

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

    print('Model loaded correctly')

    return layers