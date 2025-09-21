import numpy as np
import DnnLib
import json

train_data = np.load('mnist_train.npz')
test_data = np.load('mnist_test.npz')

images = train_data['images'].reshape(train_data['images'].shape[0], -1) / 255
labels = train_data['labels']

test_images = test_data['images'].reshape(test_data['images'].shape[0], -1) / 255
test_labels = test_data['labels']

test_one_hot = []
for label in test_labels:
    out = [0] * 10
    out[label] = 1
    test_one_hot.append(out)
test_one_hot = np.array(test_one_hot) 

N = images.shape[0]
val_size = int(N * 0.1)

idx = np.arange(N)
np.random.shuffle(idx)

val_idx, train_idx = idx[:val_size], idx[val_size:] 

val_images, val_labels = images[val_idx], labels[val_idx]
train_images, train_labels = images[train_idx], labels[train_idx]

val_one_hot = []
for label in val_labels:
    out = [0] * 10
    out[label] = 1
    val_one_hot.append(out)
val_one_hot = np.array(val_one_hot) 

print(f"\n valuation shape: {val_images.shape}, train shape {train_images.shape}")

layer1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
layer2 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)

optimizer = DnnLib.Adam(learning_rate=0.001)

N = train_images.shape[0]

epochs, batch_size = 15, 216

for epoch in range(1, epochs + 1):
    epoch_loss, epoch_acc = [], []
    
    for i in range(0, N, batch_size):
        batch = []
        batch_labels = []
        
        start_idx = i
        end_idx = min(i + batch_size, N)
        
        batch = train_images[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx]

        one_hot_classes = []

        for i in batch_labels:
            out = [0] * 10
            out[i] = 1
            one_hot_classes.append(out)
        
        one_hot_classes = np.array(one_hot_classes)

        out1 = layer1.forward(batch)
        out2 = layer2.forward(out1)

        loss = DnnLib.cross_entropy(out2, one_hot_classes)
        predictions = np.argmax(out2, axis=1)
        accuracy = (predictions == np.argmax(one_hot_classes, axis=1)).mean()
        
        epoch_loss.append(loss)
        epoch_acc.append(accuracy)
        
        scce_grad = DnnLib.cross_entropy_gradient(out2, one_hot_classes)
        grad1 = layer2.backward(scce_grad)
        grad2 = layer1.backward(grad1)

        optimizer.update(layer2)
        optimizer.update(layer1)

    avg_loss = np.array(epoch_loss).mean()
    avg_acc = np.array(epoch_acc).mean()
    print(f"Epoch (Training) # {epoch} | Loss : {avg_loss} | Accuracy: {avg_acc}")   

    out_test1 = layer1.forward(val_images)
    out_test2 = layer2.forward(out_test1)
    
    val_loss = DnnLib.cross_entropy(out_test2, val_one_hot)
    val_predictions = np.argmax(out_test2, axis=1)
    val_accuracy = (val_predictions == np.argmax(val_one_hot, axis=1)).mean()
    print(f"Epoch (Test)     # {epoch} | Loss : {val_loss} | Accuracy: {(val_accuracy * 100):.2f}%")
    
out_test1 = layer1.forward(test_images)
out_test2 = layer2.forward(out_test1)
    
test_loss = DnnLib.cross_entropy(out_test2, test_one_hot)
test_predictions = np.argmax(out_test2, axis=1)
test_accuracy = (test_predictions == np.argmax(test_one_hot, axis=1)).mean()

print("Model test with mnist_test.npz dataset")
print(f"Model Test | Accuracy: {(test_accuracy * 100):.2f}%")

# 5. Guardar el modelo
    # Guardar el modelo en un archivo json con el formato del ing

input_shape = [28, 28]
preprocess = { "scale": 255.0 }

layer1_data = {
    "type": "dense",
    "units": len(layer1.bias),
    "activation": str(layer1.activation_type).split(".")[1].lower(),
    "W": layer1.weights.T.tolist(),
    "b": layer1.bias.tolist()
}

layer2_data = {
    "type": "dense",
    "units": len(layer2.bias),
    "activation": str(layer2.activation_type).split(".")[1].lower(),
    "W": layer2.weights.T.tolist(),
    "b": layer2.bias.tolist()
}

model = {
    "input_space": input_shape,
    "preprocess": preprocess,
    "layers": [
        layer1_data,
        layer2_data
    ]
}

with open('mnist_model_v1.json', 'w') as f:
    json.dump(model, f)
    
    
with open('mnist_mlp_pretty.json', 'r') as f:
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

    print(new_layer.weights.shape)
    print(new_layer.bias.shape)
    
    new_layer.weights = W.T
    new_layer.bias = b

    layers.append(new_layer)

out_test1 = layers[0].forward(test_images)
out_test2 = layers[1].forward(out_test1)
    
test_loss = DnnLib.cross_entropy(out_test2, test_one_hot)
test_predictions = np.argmax(out_test2, axis=1)
test_accuracy = (test_predictions == np.argmax(test_one_hot, axis=1)).mean()

print("Model test with mnist_test.npz dataset")
print(f"Model Test | Accuracy: {(test_accuracy * 100):.2f}%")