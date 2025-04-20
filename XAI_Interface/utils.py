import numpy as np
import copy
import matplotlib
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'




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




# --------------------------------------
# Load parameters
# --------------------------------------

def loadparams():
    W = [np.loadtxt('new_params_last/l%d-W.txt'%l) for l in range(1,4)]
    B = [np.loadtxt('new_params_last/l%d-B.txt'%l) for l in range(1,4)]
    return W,B

def clean_loadparams():
    W = [np.loadtxt('clean_new_params_last/l%d-W.txt'%l) for l in range(1,4)]
    B = [np.loadtxt('clean_new_params_last/l%d-B.txt'%l) for l in range(1,4)]
    return W,B



def loaddata(index, data_type='test'):
    """ Load a single image and label from the MNIST dataset.
    """
    from torchvision import datasets, transforms
    import numpy as np

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    is_train = data_type == 'train'
    mnist_dataset = datasets.MNIST(root='./', train=is_train, download=True, transform=transform)

    # Extract single image (X) and label (T)
    X = mnist_dataset.data[index].numpy() / 255.0  # Normalize to [0, 1]
    T = mnist_dataset.targets[index].item()        # Single label as int

    return X, T


def loaddata_clean(index, data_type='test', data_dir="clean_data_last"):
    """ Load a single image and label from the cleaned MNIST dataset stored in clean_data_last directory.
    """
    import os
    import struct
    import numpy as np

    # Determine file paths
    if data_type == 'train':
        images_path = os.path.join(data_dir, "train-images-idx3-ubyte")
        labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte")
    elif data_type == 'test':
        images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte")
        labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte")
    else:
        raise ValueError("data_type must be 'train' or 'test'.")

    # Load labels
    with open(labels_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in label file. Expected 2049.")
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    # Load images
    with open(images_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in image file. Expected 2051.")
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)

    # Validate the index
    if index < 0 or index >= len(labels):
        raise IndexError(f"Index {index} out of range for dataset of size {len(labels)}.")

    # Extract a single image and label
    image = images[index].astype(np.float32) / 255.0  # Normalize to [0, 1]
    label = int(labels[index])

    return image, label



