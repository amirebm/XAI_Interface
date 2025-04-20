import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import random
import utils_saves
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,  precision_score, recall_score, f1_score
import torch.nn.functional as F
from torchvision import transforms
import seaborn as sns
from torchvision import datasets, transforms
from utils import loaddata, loaddata_clean
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import struct
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



############################*****COMPUTE CLASS SCORES***********##############################
def compute_class_scores(image, label, dataset_type="original"):
    """ Compute class scores for a given image using a multi-layer neural network. """
    # Load parameters
    if dataset_type == "clean":
        W, B = utils_saves.clean_loadparams()  # Load weights and biases for clean dataset
    else:
        W, B = utils_saves.loadparams()  # Load weights and biases for original dataset

    L = len(W)  # Number of layers
    A = [image.flatten()] + [None] * L  # Initialize activations

    # Step 1: Forward pass (compute activations)
    for l in range(L):
        A[l + 1] = np.maximum(0, A[l].dot(W[l]) + B[l])  # ReLU activation
        print(f"[DEBUG] Forward Pass - Activations A[{l + 1}]: {A[l + 1]}")

    # Step 2: Initialize relevance for the specific label
    one_hot_modified = (np.arange(10) == label).astype(np.float32)  # One-hot vector for the given label
    R = [None] * L + [A[L] * one_hot_modified]  # Initialize relevance scores for the last layer
    print(f"[DEBUG] Relevance Initialization R[{L}]: {R[L]}")

    # Step 3: Backward pass to compute relevance
    for l in range(L - 1, 0, -1):
        w = W[l]
        b = B[l]
        z = A[l].dot(w) + b + 1e-9  # Stabilized denominator
        s = R[l + 1] / z            # Element-wise division
        c = s.dot(w.T)              # Backward pass
        R[l] = A[l] * c             # Element-wise multiplication

    # Step 4: Forward pass with modified relevance
    for l in range(L):
        R[l + 1] = np.maximum(0, R[l].dot(W[l]) + B[l])  # ReLU forward pass with relevance
    return R[L]  # Final raw scores for all classes

def compute_class_scores_simple(image, new_label, dataset_type="original"):
    """ Compute class scores for a given image using a multi-layer neural network.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = torch.tensor(image, dtype=torch.float32, device=device)
   
    # Flatten the image
    X = image.view(1, -1)  # Shape: (1, 784)

    # Load parameters (weights and biases for each layer)
    if dataset_type == "clean":
        W, B = utils_saves.clean_loadparams()
    else:
        W, B = utils_saves.loadparams()

    # Convert weights and biases to PyTorch tensors, set dtype to float32, and move to GPU
    W = [torch.tensor(w, dtype=torch.float32, device=device) for w in W]
    B = [torch.tensor(b, dtype=torch.float32, device=device) for b in B]

    # Forward pass
    A = X  # Initialize activation
    for l in range(len(W) - 1):  # Apply ReLU activation to hidden layers
        A = torch.relu(A.mm(W[l]) + B[l])

    # Final layer (output layer) without ReLU
    A = A.mm(W[-1]) + B[-1]

    # Return the raw scores (output of the final layer) and the new label
    return A[0], new_label

def predict_class(image, W, B):
    """ Predict the class of a given image using a multi-layer neural network."""
    activations = image.flatten()  # Flatten the input image

    # Forward pass through each layer
    for w, b in zip(W, B):
        activations = np.dot(activations, w) + b  # Linear transformation
        activations = np.maximum(0, activations)  # Apply ReLU activation

    # The final activations represent logits for each class
    return np.argmax(activations)  # Return the class with the highest score
#####################################COMPUTE CLASS SCORES############################################


####################***********OVERALL CACLCULATION OF ALL METRICS***********##############

from sklearn.metrics import confusion_matrix

def calculate_overall_metrics(data_type, dataset_size=10000, dataset_type="original"):
    """ Calculate overall metrics for the clean dataset. """
     
    if dataset_type == "clean":
        W, B = utils_saves.clean_loadparams()  # Load weights and biases for clean dataset
    else:
        W, B = utils_saves.loadparams()  # Load weights and biases for original dataset

    true_labels = []
    predicted_labels = []

    # Collect true and predicted labels
    for idx in range(dataset_size):
        image, true_label = utils_saves.loaddata_clean(idx, data_type)
        predicted_label = predict_class(image, W, B)

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Calculate TP, FP, FN, TN
    TP = np.diag(cm)  # True Positives are the diagonal elements
    FP = cm.sum(axis=0) - TP  # False Positives are column sums minus TP
    FN = cm.sum(axis=1) - TP  # False Negatives are row sums minus TP
    TN = cm.sum() - (FP + FN + TP)  # True Negatives are total samples minus (FP + FN + TP)

    # Calculate overall metrics
    precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)

    # Save metrics and confusion matrix details to a file
    save_dir = "metrics_with_confusion"
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, f"metrics_and_confusion_{data_type}.txt"), "w") as f:
        f.write(f"Overall Precision: {precision:.4f}\n")
        f.write(f"Overall Recall: {recall:.4f}\n")
        f.write(f"Overall F1-Score: {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n\n")
        f.write("Per-Class Metrics:\n")
        for i, (tp, fp, fn, tn) in enumerate(zip(TP, FP, FN, TN)):
            f.write(f"Class {i} -> TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")

    # Print confusion matrix details
    print("Confusion Matrix:\n", cm)
    for i, (tp, fp, fn, tn) in enumerate(zip(TP, FP, FN, TN)):
        print(f"Class {i} -> TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    print(f"Overall metrics saved to '{save_dir}' for {data_type} data.")
    return {"precision": precision, "recall": recall, "f1_score": f1, "confusion_matrix": cm}


 ####################**********OVERALL CACLCULATION OF ALL METRICS**********##############

##########################*********evaluating the model codes**********###########################

def evaluate_model_with_metrics(model, test_loader, device):
    """ Evaluate the model using the test dataset and calculate additional metrics. """ 
    model.eval()
    total = 0
    correct = 0
    all_labels = []
    all_predictions = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Get predicted labels and confidence scores
            _, predicted = torch.max(outputs, 1)
            confidences = torch.softmax(outputs, dim=1).max(dim=1).values

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    accuracy = 100 * correct / total
    # Compute additional metrics
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "labels": all_labels,
        "predictions": all_predictions,
        "confidences": all_confidences,
    }

def evaluate_saved_models_with_metrics(original_params_loader, clean_params_loader, batch_size=64):
    """ Evaluate the original and clean models using the test datasets and calculate metrics. """
    # Load the datasets
    original_data = load_original_data(data_dir="data/MNIST/raw")
    clean_data = load_clean_data(data_path="clean_data_last/train_clean.pkl")

    # Normalize the data
    original_images = torch.tensor(original_data["images"]).float().div(255.0).view(-1, 28 * 28)
    clean_images = torch.tensor(clean_data["images"]).float().div(255.0).view(-1, 28 * 28)

    original_labels = torch.tensor(original_data["labels"]).long()
    clean_labels = torch.tensor(clean_data["labels"]).long()

    original_dataset = TensorDataset(original_images, original_labels)
    clean_dataset = TensorDataset(clean_images, clean_labels)

    original_test_loader = DataLoader(original_dataset, batch_size=batch_size, shuffle=False)
    clean_test_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False)

    # Initialize models
    original_model = SimpleDeepRectifierNet()
    clean_model = SimpleDeepRectifierNet()

    # Load weights and biases into the models
    original_params = original_params_loader()
    clean_params = clean_params_loader()

    # Evaluate models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_model.to(device)
    clean_model.to(device)

    original_metrics = evaluate_model_with_metrics(original_model, original_test_loader, device)
    clean_metrics = evaluate_model_with_metrics(clean_model, clean_test_loader, device)

    # Print debug info
    print("\nOriginal Model Predictions:", np.unique(original_metrics["predictions"], return_counts=True))
    print("Clean Model Predictions:", np.unique(clean_metrics["predictions"], return_counts=True))

    
##########################*************evaluating the model codes***********###########################




###############################********CALCULATE CLASS METRICS *******############################

def calculate_class_metrics_clean(data_type, num_classes=10, dataset_size=10000, data_dir="clean_data_last"):
    """ Calculate precision, recall, and F1-score for each class and save to files.
    """
    # Load weights and biases for the clean dataset
    W, B = utils_saves.clean_loadparams()

    true_labels = []
    predicted_labels = []

    # Collect true and predicted labels
    for idx in range(dataset_size):
        # Use the loaddata_clean function to load data from the clean dataset
        image, true_label = utils_saves.loaddata_clean(idx, data_type, data_dir)
        predicted_label = predict_class(image, W, B)

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Initialize dictionaries to store metrics
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}

    # Compute metrics for each class
    for class_label in range(num_classes):
        # Create binary labels for the current class
        true_binary = (true_labels == class_label).astype(int)
        predicted_binary = (predicted_labels == class_label).astype(int)

        # Calculate precision, recall, and F1-score for the class
        precision = precision_score(true_binary, predicted_binary, zero_division=0)
        recall = recall_score(true_binary, predicted_binary, zero_division=0)
        f1 = f1_score(true_binary, predicted_binary, zero_division=0)

        precision_scores[class_label] = precision
        recall_scores[class_label] = recall
        f1_scores[class_label] = f1

    # Save metrics to files
    save_dir = "clean_class_metrics_new_last"
    os.makedirs(save_dir, exist_ok=True)

    # Save precision scores
    with open(os.path.join(save_dir, f"precision_{data_type}.txt"), "w") as f:
        for class_label, score in precision_scores.items():
            f.write(f"Class {class_label}: {score:.4f}\n")

    # Save recall scores
    with open(os.path.join(save_dir, f"recall_{data_type}.txt"), "w") as f:
        for class_label, score in recall_scores.items():
            f.write(f"Class {class_label}: {score:.4f}\n")

    # Save F1-scores
    with open(os.path.join(save_dir, f"f1_score_{data_type}.txt"), "w") as f:
        for class_label, score in f1_scores.items():
            f.write(f"Class {class_label}: {score:.4f}\n")

    print(f"Metrics saved to '{save_dir}' for {data_type} data.")
    return precision_scores, recall_scores, f1_scores



def calculate_class_metrics_original(data_type, num_classes=10, dataset_size=10000,dataset_type="original"):
    """ Calculate precision, recall, and F1-score for each class and save to files.     """
    
    # Load weights and biases for the clean dataset
    if dataset_type == "clean":
        W, B = utils_saves.clean_loadparams()  # Load weights and biases for clean dataset
    else:
        W, B = utils_saves.loadparams()  # Load weights and biases for original dataset
    true_labels = []
    predicted_labels = []

    # Collect true and predicted labels
    for idx in range(dataset_size):
        image, true_label = utils_saves.loaddata(idx, data_type)
        predicted_label = predict_class(image, W, B)

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Initialize dictionaries to store metrics
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}

    # Compute metrics for each class
    for class_label in range(num_classes):
        # Create binary labels for the current class
        true_binary = (true_labels == class_label).astype(int)
        predicted_binary = (predicted_labels == class_label).astype(int)

        # Calculate precision, recall, and F1-score for the class
        precision = precision_score(true_binary, predicted_binary, zero_division=0)
        recall = recall_score(true_binary, predicted_binary, zero_division=0)
        f1 = f1_score(true_binary, predicted_binary, zero_division=0)

        precision_scores[class_label] = precision
        recall_scores[class_label] = recall
        f1_scores[class_label] = f1

    # Save metrics to files
    save_dir = "original_class_metrics_new_last"
    os.makedirs(save_dir, exist_ok=True)

    # Save precision scores
    with open(os.path.join(save_dir, f"precision_{data_type}.txt"), "w") as f:
        for class_label, score in precision_scores.items():
            f.write(f"Class {class_label}: {score:.4f}\n")

    # Save recall scores
    with open(os.path.join(save_dir, f"recall_{data_type}.txt"), "w") as f:
        for class_label, score in recall_scores.items():
            f.write(f"Class {class_label}: {score:.4f}\n")

    # Save F1-scores
    with open(os.path.join(save_dir, f"f1_score_{data_type}.txt"), "w") as f:
        for class_label, score in f1_scores.items():
            f.write(f"Class {class_label}: {score:.4f}\n")

    print(f"Metrics saved to '{save_dir}' for {data_type} data.")
    return precision_scores, recall_scores, f1_scores

###############################**********CALCULATE CLASS METRICS********############################

#################### **************TRAINING MODEL METHOD **********###################

def parse_idx(filename):
    """
    Parse IDX file format (used for MNIST) and return data as a NumPy array.
    """
    with open(filename, 'rb') as f:
        # Read the magic number and dimensions
        magic, num_items = struct.unpack(">II", f.read(8))
        if magic == 2051:  # Magic number for images
            rows, cols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.uint8).reshape(num_items, rows, cols)
        elif magic == 2049:  # Magic number for labels
            data = np.fromfile(f, dtype=np.uint8)
        else:
            raise ValueError("Invalid IDX file format")
    return data


def load_original_data(data_dir="data/MNIST/raw"):
    """
    Load the original MNIST dataset from the raw .ubyte files.
    """
    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte")

    # Parse the images and labels
    images = parse_idx(train_images_path)  
    labels = parse_idx(train_labels_path)  

    return {"images": images, "labels": labels}



def load_clean_data(data_dir="clena_data_27_02"):
    """
    Load the original MNIST dataset from the raw .ubyte files
    """
    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte")

    # Parse the images and labels
    images = parse_idx(train_images_path)  
    labels = parse_idx(train_labels_path) 

    return {"images": images, "labels": labels}

   

def train_model_last(data_type='original', epochs=20, batch_size=64, learning_rate=0.001, save_dir="model_weights"):
    """ Train a simple deep rectifier neural network on the MNIST dataset.
    """
    # Load the dataset
    if data_type == 'clean':
        data = load_clean_data(data_dir="clena_data_27_02")
    else:
        data = load_original_data(data_dir="data/MNIST/raw")

    # Prepare DataLoader
    images = torch.tensor(data["images"]).float().div(255.0).view(-1, 28 * 28)  # Normalize and flatten
    labels = torch.tensor(data["labels"]).long()

    dataset = TensorDataset(images, labels)
    train_size = int(0.8 * len(dataset))  # 80% training data
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, optimizer, and scheduler
    model = SimpleDeepRectifierNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Track best validation accuracy
    best_val_accuracy = 0.0

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Compute average loss for the epoch
        avg_loss = running_loss / len(train_loader)

        # Evaluate on the validation set
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # Compute validation loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_loss /= len(test_loader)
        val_accuracy = 100 * correct / total

        # Update the learning rate
        scheduler.step()

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

        # Log metrics
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Loss: {avg_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

    # Save the final model weights and biases in human-readable format
    save_weights_and_biases(model, save_dir=save_dir)



def load_clean_data(file_path):
    """ Load the cleaned MNIST dataset from a
    """
    with open(file_path, "rb") as f:
        images, labels = pickle.load(f)

    # Reshape images to their original 28x28 format for visualization
    images = images.reshape(-1, 28, 28)
    return images, labels


# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Logits
        return x


# Function to load MNIST data
def load_mnist_data(data_type='train'):
    transform = transforms.Compose([transforms.ToTensor()])
    is_train = data_type == 'train'
    mnist_dataset = datasets.MNIST(root='./data', train=is_train, download=True, transform=transform)
    X = mnist_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0  # Normalize and flatten
    y = mnist_dataset.targets.numpy()
    return X, y



# Define the neural network
class SimpleDeepRectifierNet(nn.Module):
    def __init__(self):
        super(SimpleDeepRectifierNet, self).__init__()
        self.fc1 = nn.Linear(784, 300)  # Input to Hidden Layer 1
        self.fc2 = nn.Linear(300, 100)  # Hidden Layer 1 to Hidden Layer 2
        self.fc3 = nn.Linear(100, 10)   # Hidden Layer 2 to Output Layer

        # Apply weight initialization
        self._initialize_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))  # ReLU Activation for Hidden Layer 1
        x = torch.relu(self.fc2(x))  # ReLU Activation for Hidden Layer 2
        x = self.fc3(x)  # Output layer (no activation; softmax is applied in loss function)
        return x

    def _initialize_weights(self):
        # Apply Xavier initialization to weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        # Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

def save_model_parameters(model, output_dir="clean_new_params_last"):
    """ Save the model parameters (weights and biases) in plain text format.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save weights and biases for each layer as plain text
    np.savetxt(os.path.join(output_dir, "l1-W.txt"), model.fc1.weight.data.numpy())
    np.savetxt(os.path.join(output_dir, "l1-B.txt"), model.fc1.bias.data.numpy())
    np.savetxt(os.path.join(output_dir, "l2-W.txt"), model.fc2.weight.data.numpy())
    np.savetxt(os.path.join(output_dir, "l2-B.txt"), model.fc2.bias.data.numpy())
    np.savetxt(os.path.join(output_dir, "l3-W.txt"), model.fc3.weight.data.numpy())
    np.savetxt(os.path.join(output_dir, "l3-B.txt"), model.fc3.bias.data.numpy())

    print(f"Model parameters saved in plain text format in '{output_dir}'")



# Function to load the cleaned training dataset
def load_clean_dataset(file_path):
    """ Load the cleaned training dataset from a pickle file.
    """
    with open(file_path, "rb") as f:
        X_clean, y_clean = pickle.load(f)
    return X_clean, y_clean


# Main function to train the model and save parameters
def train_and_save_params(clean_train_path, output_dir, epochs=20, batch_size=64, learning_rate=0.001):
    """ Train the model on the cleaned dataset and save the model parameters.
    """
    # Load the cleaned training dataset
    X_clean, y_clean = load_clean_dataset(clean_train_path)

    # Convert data to PyTorch tensors and create DataLoader
    train_data = TensorDataset(torch.tensor(X_clean, dtype=torch.float32), torch.tensor(y_clean, dtype=torch.long))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = SimpleDeepRectifierNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # Save the model parameters
    save_model_parameters(model, output_dir)


####################****************TRAINING MODEL METHOD***************** ###################

###############################*******CALCULATE AND SAVE CONFUSION MATRICES**********############################
def calculate_confusion_matrix_per_class_clean(data_type, num_classes=10, dataset_type="clean"):
    """ Calculate and save confusion matrices for each class with FP/FN breakdowns below the images. """
    if data_type == "train":
        images_path = 'clean_data_last/train-images-idx3-ubyte'
        
    elif data_type == "test":
         images_path = 'clean_data_last/t10k-images-idx3-ubyte'
        
    else:
        raise ValueError("data_type must be 'train' or 'test'.")

    with open(images_path, 'rb') as imgpath:
        _, dataset_size, _, _ = np.fromfile(imgpath, dtype=np.dtype(">i4"), count=4)  # Read dataset size

    true_labels = []
    predicted_labels = []    


    if dataset_type == "clean":
        W, B = utils_saves.clean_loadparams()  # Load weights and biases for clean dataset
    else:
        W, B = utils_saves.loadparams()  # Load weights and biases for original dataset

    true_labels = []
    predicted_labels = []

    # Collect true and predicted labels
    for idx in range(dataset_size):
        image, true_label = utils_saves.loaddata_clean(idx, data_type)
        predicted_label = predict_class(image, W, B)

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    save_dir = "clean_confusions__new_last1"
    os.makedirs(save_dir, exist_ok=True)

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(num_classes)))

    # Process each class
    for i in range(num_classes):
        # True Positives, False Positives, False Negatives, True Negatives
        TP = cm[i, i]
        FP = sum(cm[:, i]) - TP
        FN = sum(cm[i, :]) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        # Breakdown FP and FN percentages
        fp_breakdown = []
        fn_breakdown = []

        for j in range(num_classes):
            if j != i:
                # False Positives: True class is `j`, predicted class is `i`
                fp_count = cm[j, i]
                if FP > 0:
                    fp_breakdown.append(f"{j}->{(fp_count / FP) * 100:.2f}%")

                # False Negatives: True class is `i`, predicted class is `j`
                fn_count = cm[i, j]
                if FN > 0:
                    fn_breakdown.append(f"{j}->{(fn_count / FN) * 100:.2f}%")

        # Plot the confusion matrix
        matrix = np.array([[TP, FP], [FN, TN]])
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=False,  # Remove x-axis tick labels
            yticklabels=False,  # Remove y-axis tick labels
        )
        plt.title(f"Confusion Matrix for Class {i} ({dataset_type}_{data_type.capitalize()} Data)")

        # Add FP/FN breakdowns as bold text below the heatmap
        fp_text = f"**False Positives:** {'  '.join(fp_breakdown)}" if fp_breakdown else "**False Positives:** None"
        fn_text = f"**False Negatives:** {'  '.join(fn_breakdown)}" if fn_breakdown else "**False Negatives:** None"
        plt.figtext(0.5, -0.05, fp_text, wrap=True, horizontalalignment='center', fontsize=10, fontweight="bold")
        plt.figtext(0.5, -0.1, "", wrap=True, horizontalalignment='center', fontsize=10)  # Add spacing
        plt.figtext(0.5, -0.15, fn_text, wrap=True, horizontalalignment='center', fontsize=10, fontweight="bold")

        # Save the image
        matrix_path = os.path.join(save_dir, f"confusion_matrix_class_{i}_{dataset_type}_{data_type}.png")
        plt.savefig(matrix_path, bbox_inches="tight")
        plt.close()

    print(f"Confusion matrix heatmaps with FP/FN breakdowns saved to '{save_dir}' for {data_type} data.")


def calculate_confusion_matrix_per_class_original(data_type, num_classes=10, dataset_size=9500, dataset_type="clean"):
    """ Calculate and save confusion matrices for each class with FP/FN breakdowns below the images. """
    if dataset_type == "clean":
        W, B = utils_saves.clean_loadparams()  # Load weights and biases for clean dataset
    else:
        W, B = utils_saves.loadparams()  # Load weights and biases for original dataset

    true_labels = []
    predicted_labels = []

    # Collect true and predicted labels
    for idx in range(dataset_size):
        image, true_label = utils_saves.loaddata(idx, data_type)
        predicted_label = predict_class(image, W, B)

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    save_dir = "original_confusions_new_last1"
    os.makedirs(save_dir, exist_ok=True)

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(num_classes)))

    # Process each class
    for i in range(num_classes):
        # True Positives, False Positives, False Negatives, True Negatives
        TP = cm[i, i]
        FP = sum(cm[:, i]) - TP
        FN = sum(cm[i, :]) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        # Breakdown FP and FN percentages
        fp_breakdown = []
        fn_breakdown = []

        for j in range(num_classes):
            if j != i:
                # False Positives: True class is `j`, predicted class is `i`
                fp_count = cm[j, i]
                if FP > 0:
                    fp_breakdown.append(f"{j}->{(fp_count / FP) * 100:.2f}%")

                # False Negatives: True class is `i`, predicted class is `j`
                fn_count = cm[i, j]
                if FN > 0:
                    fn_breakdown.append(f"{j}->{(fn_count / FN) * 100:.2f}%")

        # Plot the confusion matrix
        matrix = np.array([[TP, FP], [FN, TN]])
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=False,  # Remove x-axis tick labels
            yticklabels=False,  # Remove y-axis tick labels
        )
        plt.title(f"Confusion Matrix for Class {i} ({dataset_type}_{data_type.capitalize()} Data)")

        # Add FP/FN breakdowns as bold text below the heatmap
        fp_text = f"**False Positives:** {'  '.join(fp_breakdown)}" if fp_breakdown else "**False Positives:** None"
        fn_text = f"**False Negatives:** {'  '.join(fn_breakdown)}" if fn_breakdown else "**False Negatives:** None"
        plt.figtext(0.5, -0.05, fp_text, wrap=True, horizontalalignment='center', fontsize=10, fontweight="bold")
        plt.figtext(0.5, -0.1, "", wrap=True, horizontalalignment='center', fontsize=10)  # Add spacing
        plt.figtext(0.5, -0.15, fn_text, wrap=True, horizontalalignment='center', fontsize=10, fontweight="bold")

        # Save the image
        matrix_path = os.path.join(save_dir, f"confusion_matrix_class_{i}_{dataset_type}_{data_type}.png")
        plt.savefig(matrix_path, bbox_inches="tight")
        plt.close()

    print(f"Confusion matrix heatmaps with FP/FN breakdowns saved to '{save_dir}' for {data_type} data.")


###############################*******CALCULATE AND SAVE CONFUSION MATRICES**********############################


################### MISSCLASSIFIED BAR GRAPH for each class #######################
def collect_misclassified_confidences_by_class(loaddata_fn, data_type, dataset_type, num_samples):
    """ Collect confidence values for misclassified examples, grouped by class. """
    misclassified_confidences_by_class = {i: [] for i in range(10)}  # Initialize dictionary for 10 classes

    for idx in range(num_samples):
        # Load data
        image, label = loaddata_fn(idx, data_type)

        # Compute raw scores for all classes
        #raw_scores, _ = compute_class_scores(image, label, dataset_type=dataset_type)
        raw_scores, _ =compute_class_scores_simple(image, label, dataset_type=dataset_type)
        # Calculate softmax for confidence values
        exp_scores = np.exp(raw_scores)
        softmax_scores = exp_scores / np.sum(exp_scores)

        # Predicted class
        predicted_label = np.argmax(softmax_scores)

        # Check if the prediction is incorrect
        if predicted_label != label:
            # Append the confidence of the predicted (but incorrect) class to the corresponding class
            misclassified_confidences_by_class[label].append(softmax_scores[predicted_label])

    return misclassified_confidences_by_class


def visualize_misclassified_confidence_distribution_by_class():
    """
    Visualize the confidence value distributions for misclassified examples, grouped by class.
    """
    # Number of samples
    num_train_samples = 60000  # Total training samples in MNIST
    num_test_samples = 10000   # Total test samples in MNIST

    test_confidences_original = collect_misclassified_confidences_by_class(loaddata, 'test', 'original', num_test_samples)
    test_confidences_clean = collect_misclassified_confidences_by_class(loaddata_clean, 'test', 'clean', 10000)
    test_misclassified_original = {i: len(test_confidences_original[i]) for i in range(10)}
    test_misclassified_clean = {i: len(test_confidences_clean[i]) for i in range(10)}
    
    for class_label in range(10):
        test_histogram_path = plot_confidence_bargraph(
            test_confidences_original[class_label], test_confidences_clean[class_label],
            f"Test Dataset: Misclassified Confidence Value Distribution for Class {class_label} (4% bins)",
            f"test_misclassified_confidence_distribution_class_{class_label}",
            test_misclassified_original[class_label], test_misclassified_clean[class_label]
        )
    print("Test misclassified data collection finished")
    # Collect confidence values for misclassified examples, grouped by class
    train_confidences_original = collect_misclassified_confidences_by_class(loaddata, 'train', 'original', num_train_samples)
    train_confidences_clean = collect_misclassified_confidences_by_class(loaddata_clean, 'train', 'clean', 56593)

    

    # Calculate the number of misclassified examples for each class
    train_misclassified_original = {i: len(train_confidences_original[i]) for i in range(10)}
    train_misclassified_clean = {i: len(train_confidences_clean[i]) for i in range(10)}
    

    # Plot and save bar graphs with annotations for each class
    for class_label in range(10):
        train_histogram_path = plot_confidence_bargraph(
            train_confidences_original[class_label], train_confidences_clean[class_label],
            f"Train Dataset: Misclassified Confidence Value Distribution for Class {class_label} (4% bins)",
            f"train_misclassified_confidence_distribution_class_{class_label}_4_percent_bins.png",
            train_misclassified_original[class_label], train_misclassified_clean[class_label]
        )

    print("Train misclassified data collection finished")


    print("All bar graphs generated and saved!")
    return train_histogram_path, test_histogram_path

###################************ MISSCLASSIFIED BAR GRAPH for each class**********#######################

 

#############################********Calculation confidence drop WRT PGD Attacks************####################

def set_model_params(model, W, B):
    """ Set the weights and biases of the model using the provided weights and biases. """  
    with torch.no_grad():
        for i, (w, b) in enumerate(zip(W, B)):
            # Convert numpy arrays to PyTorch tensors
            w_tensor = torch.tensor(w.T, dtype=torch.float32)  # Transpose the weights
            b_tensor = torch.tensor(b, dtype=torch.float32)

            # Get model parameters
            model_layers = list(model.parameters())  # Get all model parameters

            # Debug: Print expected vs. actual shape
            print(f"Layer {i}: Expected weight shape {model_layers[2 * i].shape}, Loaded weight shape {w_tensor.shape}")
            print(f"Layer {i}: Expected bias shape {model_layers[2 * i + 1].shape}, Loaded bias shape {b_tensor.shape}")

            # Ensure shape compatibility before copying
            if model_layers[2 * i].shape == w_tensor.shape:
                model_layers[2 * i].data.copy_(w_tensor)  # Assign transposed weights
            else:
                print(f"Error: Shape mismatch for layer {i}, skipping weight assignment.")

            if model_layers[2 * i + 1].shape == b_tensor.shape:
                model_layers[2 * i + 1].data.copy_(b_tensor)  # Assign biases
            else:
                print(f"Error: Shape mismatch for layer {i}, skipping bias assignment.")


def load_model_parameters(model, W, B):
    """  """
    print("Loading Model Parameters:")
    print(f"fc1: Weights Shape {np.array(W[0]).shape}, Biases Shape {np.array(B[0]).shape}")
    print(f"fc2: Weights Shape {np.array(W[1]).shape}, Biases Shape {np.array(B[1]).shape}")
    print(f"fc3: Weights Shape {np.array(W[2]).shape}, Biases Shape {np.array(B[2]).shape}")

    # Transpose weights if they are in the wrong format
    if W[0].shape != (300, 784):
        print("Transposing fc1 weights...")
        W[0] = np.array(W[0]).T
    if W[1].shape != (100, 300):
        print("Transposing fc2 weights...")
        W[1] = np.array(W[1]).T
    if W[2].shape != (10, 100):
        print("Transposing fc3 weights...")
        W[2] = np.array(W[2]).T

    # Assign weights and biases
    model.fc1.weight.data = torch.tensor(W[0], dtype=torch.float32)
    model.fc1.bias.data = torch.tensor(B[0], dtype=torch.float32)
    model.fc2.weight.data = torch.tensor(W[1], dtype=torch.float32)
    model.fc2.bias.data = torch.tensor(B[1], dtype=torch.float32)
    model.fc3.weight.data = torch.tensor(W[2], dtype=torch.float32)
    model.fc3.bias.data = torch.tensor(B[2], dtype=torch.float32)

    print("Model parameters successfully loaded.")


# --- Load Test Data ---
def load_test_data():
    test_indices = range(100)  # Load first 100 samples for simplicity
    original_data = [utils_saves.loaddata(i) for i in test_indices]
    clean_data = [utils_saves.loaddata_clean(i) for i in test_indices]
    return original_data, clean_data

# --- Evaluation ---
def evaluate_model(model, data, attack=None):
    clean_accuracy = 0
    adversarial_accuracy = 0
    perturbations = []

    for image, label in data:
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        label_tensor = torch.tensor([label])

        # Evaluate clean accuracy
        clean_output = model(image_tensor)
        clean_pred = torch.argmax(clean_output, dim=1).item()
        clean_accuracy += (clean_pred == label)

        # Generate adversarial example
        if attack:
            adv_image = attack(model, image_tensor, label_tensor)
            adv_output = model(torch.tensor(adv_image).unsqueeze(0).unsqueeze(0))
            adv_pred = torch.argmax(adv_output, dim=1).item()
            adversarial_accuracy += (adv_pred == label)

            # Calculate perturbation magnitude (L2 Norm)
            perturbations.append(np.linalg.norm((adv_image - image).flatten()))

    clean_accuracy /= len(data)
    adversarial_accuracy /= len(data)
    avg_perturbation = np.mean(perturbations) if perturbations else 0

    return clean_accuracy, adversarial_accuracy, avg_perturbation


def pgd_attack(model, images, labels, epsilon=0.3, alpha=2/255, iters=40):
    """    Generate adversarial examples using the PGD attack.
    """   
    # Store the original images for clipping
    original_images = images.clone().detach()
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)

        # Compute loss and gradients
        model.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Apply gradient ascent step
        adv_images = images + alpha * images.grad.sign()

        # Clip perturbations to epsilon-ball around original images
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)

        # Update images with valid pixel range
        images = torch.clamp(original_images + eta, min=0, max=1).detach_()

    return images


def evaluate_model_confidence_drop_original(model, loaddata, loaddata_clean, attack_fn, attack_strengths):
    """ Evaluate the model's confidence drop under increasing attack strength for original dataset  """

    model.eval()

    original_confidence_drop = []

    for strength in attack_strengths:
        print(f"Evaluating attack strength: {strength}")
        random_indices = random.sample(range(1000), 1000)  # Generate 1000 unique random numbers in range [0, 999]
        # Confidence for Original Dataset
        original_confidence = []
        for i in random_indices:  # Use a smaller subset for visualization (adjust as needed)
            X, T = loaddata(i, data_type='test')
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            T_tensor = torch.tensor([T])

            # Generate adversarial example
            adv_X = attack_fn(model, X_tensor, T_tensor, epsilon=strength)

            # Model prediction confidence
            with torch.no_grad():
                clean_output = model(X_tensor.view(1, -1))
                adv_output = model(adv_X.view(1, -1))

                clean_confidence = F.softmax(clean_output, dim=1).max().item()
                adv_confidence = F.softmax(adv_output, dim=1).max().item()

                confidence_drop = clean_confidence - adv_confidence

            original_confidence.append(confidence_drop)

        original_confidence_drop.append(np.mean(original_confidence))

        

    # Return results
    return {
        "original": original_confidence_drop
    }



def evaluate_model_confidence_drop_clean(model, loaddata, loaddata_clean, attack_fn, attack_strengths):
    """ Evaluate the model's confidence drop under increasing attack strength for clean dataset """ 
    model.eval()

    clean_confidence_drop = []

    for strength in attack_strengths:
        print(f"Evaluating attack strength: {strength}")
        
        
        # Confidence for Clean Dataset
        clean_confidence = []
        random_indices = random.sample(range(1000), 1000)  # Generate 1000 unique random numbers in range [0, 999]

        for i in random_indices:  # Use a smaller subset for visualization (adjust as needed)
            X, T = loaddata_clean(i, data_type='test')
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            T_tensor = torch.tensor([T])

            # Generate adversarial example
            adv_X = attack_fn(model, X_tensor, T_tensor, epsilon=strength)

            # Model prediction confidence
            with torch.no_grad():
                clean_output = model(X_tensor.view(1, -1))
                adv_output = model(adv_X.view(1, -1))

                clean_confidence_clean = F.softmax(clean_output, dim=1).max().item()
                adv_confidence_clean = F.softmax(adv_output, dim=1).max().item()

                confidence_drop_clean = clean_confidence_clean - adv_confidence_clean

            clean_confidence.append(confidence_drop_clean)

        clean_confidence_drop.append(np.mean(clean_confidence))

    # Return results
    return {
        
        "clean": clean_confidence_drop
    }

def save_plot(attack_strengths, results, output_path):
    """ Save a plot of confidence drop under increasing attack strength.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.figure(figsize=(8, 6))
    plt.plot(attack_strengths, results["clean"], label=" Clean Dataset", marker="o")
    plt.plot(attack_strengths, results["original"], label="Original Dataset", marker="x")
    plt.xlabel("Attack Strength (Epsilon)")
    plt.ylabel("Average Confidence")
    plt.title("Confidence Drop Under Increasing Attack Strength")
    plt.legend()
    plt.grid()

    # Save the plot
    plot_file = os.path.join(output_path, "confidence_drop_plot.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved at: {plot_file}")


#############################********Calculation confidence drop WRT PGD Attacks************####################




###################***********comparing model performance on original and  clean data**********###############
def check_data_sizes():
    """ Check the number of samples in the original and clean datasets. """
    # Original dataset test size
    num_original_samples = 0
    try:
        while True:
            X, T = utils_saves.loaddata(num_original_samples, data_type='test')
            num_original_samples += 1
    except IndexError:
        print(f"Total Original Test Dataset Samples: {num_original_samples}")
    
    # Clean dataset test size
    num_clean_samples = 0
    try:
        while True:
            X, T = utils_saves.loaddata_clean(num_clean_samples, data_type='test')
            num_clean_samples += 1
    except IndexError:
        print(f"Total Clean Test Dataset Samples: {num_clean_samples}")


def evaluate_accuracy(model, loaddata, num_samples, dataset_name):
    """ Evaluate the model's accuracy on the specified dataset. """
    correct_predictions = 0
    for i in range(num_samples):
        X, T = loaddata(i, data_type='test')
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(X_tensor.view(1, -1))
            pred = torch.argmax(output, dim=1).item()

        correct_predictions += (pred == T)
    
    accuracy = correct_predictions / num_samples
    print(f"Accuracy on {dataset_name} Dataset: {accuracy * 100:.2f}%")
    return accuracy

def evaluate_confidence(model, loaddata, num_samples, dataset_name):
    """ Evaluate the model's average confidence on the specified dataset. """
    confidences = []
    for i in range(num_samples):
        X, T = loaddata(i, data_type='test')
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(X_tensor.view(1, -1))
            confidence = F.softmax(output, dim=1).max().item()
            confidences.append(confidence)
    
    avg_confidence = sum(confidences) / num_samples
    print(f"Average Confidence on {dataset_name} Dataset: {avg_confidence:.4f}")
    return avg_confidence


def count_clean_train_data(clean_train_path):
    """
    Count the number of samples in the cleaned training dataset."""
    # Load the cleaned training dataset
    with open(clean_train_path, "rb") as f:
        X_clean, y_clean = pickle.load(f)

    # Count the number of samples
    num_samples = X_clean.shape[0]
    return num_samples

###################***********comparing model performance on original and  clean data**********###############


###########***********TRAINING MODEL BY NEW CLEAN DATA AND WITHout CROSS VALIDATION*****##############

# Define the neural network
class SimpleDeepRectifierNet(nn.Module):
    def __init__(self):
        super(SimpleDeepRectifierNet, self).__init__()
        self.fc1 = nn.Linear(784, 300)  # Input to Hidden Layer 1
        self.fc2 = nn.Linear(300, 100)  # Hidden Layer 1 to Hidden Layer 2
        self.fc3 = nn.Linear(100, 10)   # Hidden Layer 2 to Output Layer

        # Apply weight initialization
        self._initialize_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))  # ReLU Activation for Hidden Layer 1
        x = torch.relu(self.fc2(x))  # ReLU Activation for Hidden Layer 2
        x = self.fc3(x)  # Output layer (no activation; softmax is applied in loss function)
        return x

    def _initialize_weights(self):
        # Apply Xavier initialization to weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        # Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)


def save_weights_and_biases(model, save_dir):
    """ Save the weights and biases of the model to plain text files
    """
    os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist
    fmt = "%.3f"  # Save with three decimal places

    np.savetxt(os.path.join(save_dir, "l1-W.txt"), model.fc1.weight.data.numpy().T, fmt=fmt)
    np.savetxt(os.path.join(save_dir, "l1-B.txt"), model.fc1.bias.data.numpy(), fmt=fmt)
    np.savetxt(os.path.join(save_dir, "l2-W.txt"), model.fc2.weight.data.numpy().T, fmt=fmt)
    np.savetxt(os.path.join(save_dir, "l2-B.txt"), model.fc2.bias.data.numpy(), fmt=fmt)
    np.savetxt(os.path.join(save_dir, "l3-W.txt"), model.fc3.weight.data.numpy().T, fmt=fmt)
    np.savetxt(os.path.join(save_dir, "l3-B.txt"), model.fc3.bias.data.numpy(), fmt=fmt)

    print(f"Weights and biases saved to folder '{save_dir}'")


# Function to load the original MNIST dataset
def load_original_data(data_type='train'):
    transform = transforms.Compose([transforms.ToTensor()])
    is_train = data_type == 'train'
    mnist_dataset = datasets.MNIST(root='./data', train=is_train, download=True, transform=transform)
    X = mnist_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0  # Normalize to [0, 1]
    y = mnist_dataset.targets.numpy()
    return X, y


# Function to load the cleaned dataset
def load_clean_data(data_type='train', data_dir="clean_data_last"):
    if data_type == 'train':
        data_path = os.path.join(data_dir, "train_clean.pkl")
    elif data_type == 'test':
        data_path = os.path.join(data_dir, "test.pkl")
    else:
        raise ValueError("data_type must be 'train' or 'test'.")

    with open(data_path, "rb") as f:
        images, labels = pickle.load(f)
    return images, labels



###########***********TRAINING MODEL BY NEW CLEAN DATA AND WITHout CROSS VALIDATION*****##############



###################*********MISSCLASSIFIED BAR GRAPH for whole model***********#######################



def plot_confidence_bargraph_dynamicY(conf_adv, conf_clean, title, filename, misclassified_origin, misclassified_clean):
    """ Plot and save bar graphs for confidence values with dynamic y-axis limits and annotations. """  
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Create bins for the bar graph
    bins = np.linspace(0, 1, 26)  # 25 bins between 0 and 1 (4% bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Center of each bin

    # Calculate histogram counts
    counts_clean, _ = np.histogram(conf_clean, bins=bins)
    counts_adv, _ = np.histogram(conf_adv, bins=bins)

    # Determine the maximum value among the counts for dynamic y-axis limit
    max_count = max(counts_clean.max(), counts_adv.max())

    # Plot bar graph
    plt.figure()
    bar_width = 0.01  # Thinner bars (half of the previous width)

    # Plot "Clean" and "Original" bars side by side
    plt.bar(bin_centers - bar_width / 2, counts_clean, width=bar_width, alpha=0.5, label='Clean', color='blue')
    plt.bar(bin_centers + bar_width / 2, counts_adv, width=bar_width, alpha=0.5, label='Original', color='orange')

    # Set x-axis ticks to include 0.1, 0.2, ..., 1.0
    plt.xticks(np.arange(0, 1.1, 0.1))

    # Add labels and title
    plt.title(title)
    plt.xlabel('Confidence Value')
    plt.ylabel('Frequency')
    plt.ylim(0, max_count + 5)  # Set y-axis limits dynamically with some padding

    # Move legend outside the graph to the top right
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the graph

    # Add text annotations for misclassified counts
    plt.text(0.5, -0.25, f"Overall misclassified (Original): {misclassified_origin}", 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.5, -0.30, f"Overall misclassified (Clean): {misclassified_clean}", 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)

    # Ensure directory exists
    output_dir = os.path.join("statistics", "conf_MISS4")
    os.makedirs(output_dir, exist_ok=True)

    # Save the bar graph
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight')  # Ensure annotations are included
    plt.close()

    print(f"Bar graph saved at {output_path}")

def collect_misclassified_confidences(loaddata_fn, data_type, dataset_type, num_samples):
    """ Collect confidence values for misclassified examples. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    misclassified_confidences = []

    for idx in range(num_samples):
        # Load data
        image, label = loaddata_fn(idx, data_type)

        # Move the image and label to the device
        image = torch.tensor(image, device=device, dtype=torch.float32)  # Ensure image is a tensor on the device
        label = torch.tensor(label, device=device)  # Ensure label is on the device

        # Compute raw scores for all classes on GPU
        raw_scores, _ = compute_class_scores_simple(image, label, dataset_type=dataset_type)

        # `raw_scores` is already a tensor from compute_class_scores_simple, so no need to convert again
        exp_scores = torch.exp(raw_scores)
        softmax_scores = exp_scores / torch.sum(exp_scores)

        # Predicted class
        predicted_label = torch.argmax(softmax_scores).item()

        # Check if the prediction is incorrect
        if predicted_label != label.item():  # Compare using .item() for scalar tensors
            # Append the confidence of the predicted (but incorrect) class
            misclassified_confidences.append(softmax_scores[predicted_label].item())

    return misclassified_confidences


def plot_confidence_bargraph(conf_adv, conf_clean, title, filename, misclassified_origin, misclassified_clean):
    """ Plot and save bar graphs for confidence values.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Create bins for the bar graph
    bins = np.linspace(0, 1, 26)  # 25 bins between 0 and 1 (4% bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Center of each bin

    # Calculate histogram counts
    ####calculates the number of occurance of confindence values######
    counts_clean, _ = np.histogram(conf_clean, bins=bins)
    counts_adv, _ = np.histogram(conf_adv, bins=bins)

    # Plot bar graph
    plt.figure()
    bar_width = 0.01  # Thinner bars (half of the previous width)

    # Plot "Clean" and "Original" bars side by side
    plt.bar(bin_centers - bar_width / 2, counts_clean, width=bar_width, alpha=0.5, label='Clean', color='blue')
    plt.bar(bin_centers + bar_width / 2, counts_adv, width=bar_width, alpha=0.5, label='Original', color='orange')

    # Set x-axis ticks to include 0.1, 0.2, ..., 1.0
    plt.xticks(np.arange(0, 1.1, 0.1))

    # Add labels and title
    plt.title(title)
    plt.xlabel('Confidence Value')
    plt.ylabel('Frequency')
    plt.ylim(0, 20)  # Set y-axis limits to fixed range (0 to 20)

    # Move legend outside the graph to the top right
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the graph

    # Add text annotations for misclassified counts
    plt.text(0.5, -0.25, f"Overall misclassified (Original): {misclassified_origin}", 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.5, -0.30, f"Overall misclassified (Clean): {misclassified_clean}", 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)

    # Ensure directory exists
    output_dir = os.path.join("statistics", "conf_MISS3")
    os.makedirs(output_dir, exist_ok=True)

    # Save the bar graph
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight')  # Ensure annotations are included
    plt.close()

    print(f"Bar graph saved at {output_path}")

def visualize_misclassified_confidence_distribution():
    """
    Visualize the confidence value distributions for misclassified examples.
    """
    # Number of samples
    num_train_samples = 60000  # Total training samples in MNIST
    num_test_samples = 10000    # Total test samples in MNIST

    
    # Collect confidence values for misclassified examples (test datasets)
    test_confidences_original = collect_misclassified_confidences(loaddata, 'test', 'original', num_test_samples)
    test_confidences_clean = collect_misclassified_confidences(loaddata_clean, 'test', 'clean', 10000)
    # Collect confidence values for misclassified examples (train datasets)
    
    test_misclassified_original = len(test_confidences_original)
    test_misclassified_clean = len(test_confidences_clean)

    print("Test misclassified data collection finished")
    test_histogram_path = plot_confidence_bargraph_dynamicY(
        test_confidences_original, test_confidences_clean,
        "Test Dataset: Misclassified Confidence Value Distribution (4% bins)",
        "test_misclassified_confidence_distribution_4_percent_bins.png",
        test_misclassified_original, test_misclassified_clean
    )


    train_confidences_original = collect_misclassified_confidences(loaddata, 'train', 'original', num_train_samples)
    train_confidences_clean = collect_misclassified_confidences(loaddata_clean, 'train', 'clean', 56779)
    # Calculate the number of misclassified examples
    train_misclassified_original = len(train_confidences_original)
    train_misclassified_clean = len(train_confidences_clean)
   

    # Plot and save histograms with annotations
    print("Train misclassified data collection finished")
    train_histogram_path = plot_confidence_bargraph_dynamicY(
        train_confidences_original, train_confidences_clean,
        "Train Dataset: Misclassified Confidence Value Distribution (4% bins)",
        "train_misclassified_confidence_distribution_4_percent_bins.png",
        train_misclassified_original, train_misclassified_clean
    )

    
    return train_histogram_path, test_histogram_path


###################************MISSCLASSIFIED BAR GRAPH for whole model***********#######################


