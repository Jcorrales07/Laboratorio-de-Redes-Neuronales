import DnnLib
from utils import get_data_batch, get_accuracy, get_epoch_metrics, model_evaluation

def build_model(learning_rate=0.001):
    layers = [
        DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU),
        DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    ]

    optimizer = DnnLib.Adam(learning_rate=learning_rate)
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
        
def train_model(layers, n_samples, optimizer, train_data, val_data, epochs=15, batch_size=216):
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