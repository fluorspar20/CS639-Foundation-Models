import struct

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nn import NeuralNetwork

def train_model(X_train, y_train, X_test, y_test, hidden_units, lr, epochs=10, batch_size=32, task='classification'):
    np.random.seed(0)
    input_units = X_train.shape[1]
    output_units = y_train.shape[1]
    
    nn = NeuralNetwork(input_units, hidden_units, output_units, task=task)
    
    train_losses = []
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_losses = []
        
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # Forward pass
            y_pred = nn.forward(X_batch)
            
            # Compute loss
            loss = nn.compute_loss(y_batch, y_pred)
            epoch_losses.append(loss)
            
            # Backward pass and update weights
            nn.backward(X_batch, y_batch, y_pred, learning_rate=lr)
        
        train_losses.append(np.mean(epoch_losses))

    # print test set accuracy
    y_pred = nn.forward(X_test)
    test_loss = nn.compute_loss(y_test,y_pred)
    
    metrics = {"test_loss": test_loss}
    if task == "classification":
        acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
        metrics["accuracy"] = acc
    
    return train_losses, metrics


def run_experiment(X_train, y_train, X_test, y_test, task, dataset_name):
    print(f"\n--- Running Experiments for {dataset_name} ---")
    
    # Varying Learning Rate
    lrs = [1, 1e-2, 1e-3, 1e-8]
    plt.figure(figsize=(10, 5))
    for lr in lrs:
        losses, metrics = train_model(X_train, y_train, X_test, y_test, hidden_units=5, lr=lr, epochs=10, task=task)
        plt.plot(range(1, 11), losses, label=f'LR: {lr}')
        print(f"LR {lr}: Test Loss = {metrics['test_loss']:.4f}, Train Loss = {losses[-1]:.4f}")
        if task == "classification":
            print(f"Test Accuracy = {metrics['accuracy']:.4f}")
            
    plt.title(f'Average Training Loss vs Epochs (Varying LR) - {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.savefig(f'{dataset_name}_LR_experiment.png')
    plt.close()

    # Varying Hidden Units
    hidden_sizes = [2, 8, 16, 32]
    plt.figure(figsize=(10, 5))

    if dataset_name == "Iris":
        lr = 1
    else:        
        lr = 1e-2

    for hs in hidden_sizes:
        losses, metrics = train_model(X_train, y_train, X_test, y_test, hidden_units=hs, lr=lr, epochs=10, task=task)
        plt.plot(range(1, 11), losses, label=f'Hidden Units: {hs}')
        print(f"Hidden Units {hs}: Test Loss = {metrics['test_loss']:.4f}, Train Loss = {losses[-1]:.4f}")
        if task == "classification":
            print(f"Test Accuracy = {metrics['accuracy']:.4f}")

    plt.title(f'Average Training Loss vs Epochs (LR={lr}) - {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.savefig(f'{dataset_name}_HiddenUnits_experiment.png')
    plt.close()

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num_images, rows, cols)
        return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def main():
    # ---------------- 1. Iris Dataset-----------------
    df_iris = pd.read_csv('data/iris.csv', header=None)
    X_iris = df_iris.drop(columns=[4]).values.astype(float)
    y_iris_labels = pd.get_dummies(df_iris.iloc[:, -1]).values.astype(float)

    # standardize inputs
    X_iris = (X_iris - np.mean(X_iris, axis=0)) / np.std(X_iris, axis=0)
    
    indices = np.random.permutation(X_iris.shape[0])
    split_idx = int(X_iris.shape[0] * 0.8)
    X_iris_train, X_iris_test = X_iris[indices[:split_idx]], X_iris[indices[split_idx:]]
    y_iris_train, y_iris_test = y_iris_labels[indices[:split_idx]], y_iris_labels[indices[split_idx:]]
    
    run_experiment(X_iris_train, y_iris_train, X_iris_test, y_iris_test, "classification", "Iris")

    # ----------------- 2. California Housing Dataset -----------------
    df_housing = pd.read_csv('data/housing.csv')
    # replace missing values in total_bedrooms with mean
    df_housing['total_bedrooms'].fillna(df_housing['total_bedrooms'].mean(), inplace=True)
    # encode categorical features
    df_housing = pd.get_dummies(df_housing, columns=['ocean_proximity'])

    X_housing = df_housing.drop(columns=['median_house_value']).values.astype(float)
    y_housing = df_housing[['median_house_value']].values.astype(float)
    
    # Standardize
    X_housing = (X_housing - np.mean(X_housing, axis=0)) / np.std(X_housing, axis=0)
    y_housing = (y_housing - np.mean(y_housing)) / np.std(y_housing)
    
    indices = np.random.permutation(X_housing.shape[0])
    split_idx = int(X_housing.shape[0] * 0.8)
    X_housing_train, X_housing_test = X_housing[indices[:split_idx]], X_housing[indices[split_idx:]]
    y_housing_train, y_housing_test = y_housing[indices[:split_idx]], y_housing[indices[split_idx:]]
    
    run_experiment(X_housing_train, y_housing_train, X_housing_test, y_housing_test, "regression", "Housing")

    # ------------------ 3. MNIST Dataset ------------------
    X_train_mnist = load_mnist_images("data/train-images.idx3-ubyte")
    y_train_mnist_labels = load_mnist_labels("data/train-labels.idx1-ubyte")
    X_test_mnist = load_mnist_images("data/t10k-images.idx3-ubyte")
    y_test_mnist_labels = load_mnist_labels("data/t10k-labels.idx1-ubyte")

    X_train_mnist = X_train_mnist.reshape(X_train_mnist.shape[0], -1).astype(float) / 255.0
    X_test_mnist = X_test_mnist.reshape(X_test_mnist.shape[0], -1).astype(float) / 255.0

    y_train_mnist = np.eye(10)[y_train_mnist_labels]
    y_test_mnist = np.eye(10)[y_test_mnist_labels]

    run_experiment(X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist, "classification", "MNIST")
    

if __name__ == "__main__":
    main()