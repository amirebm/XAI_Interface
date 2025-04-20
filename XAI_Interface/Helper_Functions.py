import numpy as np
from PIL import Image
import base64
import io
import utils
import matplotlib.pyplot as plt
import streamlit as st
import torch.nn.functional as F
import streamlit as st
import cv2
from PIL import Image
from scipy.ndimage import center_of_mass, shift
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.ndimage
from PIL import Image


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras


    
##########################******* START CODE BLOCK 1********#################################

# Load random and misclassified images
NUM_IMAGES = 12


def reload_random_images(data_type='test', dataset="original", limit=11):
    """ Load a specified number of random images from the dataset. """
    import numpy as np
       # Dynamically calculate dataset size for the clean dataset
    if dataset == "clean":
            dataset_size = 10000 if data_type == 'test' else 56593
    elif dataset == "original":
            dataset_size = 10000 if data_type == 'test' else 60000
    else:
            raise ValueError("Invalid dataset type. Must be 'original' or 'clean'.")

    all_indices = np.arange(dataset_size)
    np.random.shuffle(all_indices)

    random_images = []
    random_true_labels = []
    random_predictions = []
    selected_indices = []

    for idx in all_indices:
        if len(random_images) >= limit:  # Stop when the limit is reached
            break

        try:
            # Choose which loaddata function to call
            if dataset == "clean":
                image, true_label = utils.loaddata_clean(idx, data_type=data_type)  # Use cleaned dataset
                W, B = utils.clean_loadparams()  # Load weights and biases
            else:
                image, true_label = utils.loaddata(idx, data_type=data_type)  # Use original dataset
                W, B = utils.loadparams()  # Load weights and biases

            image = image.reshape(1, -1)  # Flatten the image
            activations = image

            # Forward pass
            for w, b in zip(W, B):
                activations = np.maximum(0, np.dot(activations, w) + b)

            predicted_label = np.argmax(activations)
            random_images.append(image.reshape(28, 28))
            random_true_labels.append(true_label)
            random_predictions.append(predicted_label)
            selected_indices.append(idx)

        except IndexError as e:
            # Log and skip invalid indices for the cleaned dataset
            print(f"Skipping index {idx} for dataset '{dataset}': {e}")
            continue
    return {
        "images": random_images,
        "true_labels": random_true_labels,
        "predictions": random_predictions,
        "indices": selected_indices,
    }


_last_index = 0


# Global variable to track the last processed index
_last_index = 0

# Dataset sizes
DATASET_SIZES = {
    "clean": {"train": 56593, "test": 10000},
    "original": {"train": 60000, "test": 10000}
}

def reload_misclassified_images(data_type='test', dataset="original", limit=11, chunk_size=1000, reset=False):
    """ Load misclassified images from the dataset. """
    global _last_index

    if dataset not in DATASET_SIZES or data_type not in DATASET_SIZES[dataset]:
        raise ValueError("Invalid dataset or data_type. Must be 'original' or 'clean', and 'test' or 'train'.")

    dataset_size = DATASET_SIZES[dataset][data_type]

    if reset:
        _last_index = 0

    misclassified_images = []
    start_index = _last_index

    # Load model parameters
    W, B = (utils.clean_loadparams() if dataset == "clean" else utils.loadparams())

    for chunk_start in range(start_index, dataset_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, dataset_size)
        chunk_indices = range(chunk_start, chunk_end)

        images, true_labels = [], []

        # Load images and labels in batch
        for idx in chunk_indices:
            try:
                image, true_label = (utils.loaddata_clean(idx, data_type) if dataset == "clean" 
                                     else utils.loaddata(idx, data_type))
                images.append(image.reshape(1, -1))
                true_labels.append(true_label)
            except IndexError:
                continue

        if not images:
            continue

        images = np.vstack(images)
        true_labels = np.array(true_labels)

        # Forward pass through the model
        activations = images
        for w, b in zip(W, B):
            activations = np.maximum(0, np.dot(activations, w) + b)

        predicted_labels = np.argmax(activations, axis=1)

        # Identify misclassified images
        for idx, true_label, predicted_label, image in zip(chunk_indices, true_labels, predicted_labels, images):
            if predicted_label != true_label:
                misclassified_images.append({
                    "id": idx,
                    "image": image.reshape(28, 28),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                })

            if len(misclassified_images) >= limit:
                _last_index = idx + 1
                return misclassified_images

    _last_index = dataset_size
    return misclassified_images


##########################******* END CODE BLOCK 1********#################################


##########################******* START CODE BLOCK 2 ********#################################


def get_confidence_distribution_path(class_label, data_type):
    """ Retrieve the file path for the confidence distribution image based on the class and data type. """
    
    # Base directory for confidence distribution images
    base_dir = "Confidence_Misclassified/each_class"

    # Validate data_type
    if data_type not in ["train", "test"]:
        raise ValueError("Invalid data_type. Must be 'train' or 'test'.")

    # Build the file name based on parameters
    confidence_distribution_file = f"{data_type}_misclassified_confidence_distribution_class_{class_label}.png"

    # Combine directory and file name to get the full path
    confidence_distribution_path = os.path.join(base_dir, confidence_distribution_file)

    return confidence_distribution_path



def get_class_metrics(class_label, data_type="test", dataset_type="original"):
    """ Retrieve precision, recall, and F1-score for the specified class, data type, and dataset type. """
    
    # Determine the directory based on dataset_type
    if dataset_type == "clean":
        save_dir = "clean_class_metrics_last"
    elif dataset_type == "original":
        save_dir = "original_class_metrics_last"
    else:
        raise ValueError("Invalid dataset_type. Must be 'original' or 'clean'.")

    metrics = {}

    # Load precision
    with open(os.path.join(save_dir, f"precision_{data_type}.txt"), "r") as f:
        for line in f:
            class_num, score = line.strip().split(": ")
            if int(class_num.split()[1]) == class_label:
                metrics["precision"] = float(score)

    # Load recall
    with open(os.path.join(save_dir, f"recall_{data_type}.txt"), "r") as f:
        for line in f:
            class_num, score = line.strip().split(": ")
            if int(class_num.split()[1]) == class_label:
                metrics["recall"] = float(score)

    # Load F1-score
    with open(os.path.join(save_dir, f"f1_score_{data_type}.txt"), "r") as f:
        for line in f:
            class_num, score = line.strip().split(": ")
            if int(class_num.split()[1]) == class_label:
                metrics["f1_score"] = float(score)

    return metrics


def get_model_metrics(data_type="test", dataset_type="original"):
    """ Retrieve overall precision, recall, and F1-score for the specified data type and dataset type. """  
    
    # Determine the directory based on dataset_type
    if dataset_type == "clean":
        save_dir = "clean_metrics_overall"
    elif dataset_type == "original":
        save_dir = "original_metrics_overall"
    else:
        raise ValueError("Invalid dataset_type. Must be 'original' or 'clean'.")

    # Initialize metrics dictionary
    metrics = {}

    # Construct the file path
    file_path = os.path.join(save_dir, f"overall_metrics_{data_type}.txt")

    # Load metrics from the file
    with open(file_path, "r") as f:
        for line in f:
            if "Overall Precision" in line:
                metrics["precision"] = float(line.split(": ")[1])
            elif "Overall Recall" in line:
                metrics["recall"] = float(line.split(": ")[1])
            elif "Overall F1-Score" in line:
                metrics["f1_score"] = float(line.split(": ")[1])

    return metrics


def get_confusion_matrix_path(class_label, data_type, dataset_type="original"):
    """ Retrieve the file path for the confusion matrix image based on the class, data type, and dataset type. """  
    
    # Determine the directory based on dataset_type
    if dataset_type == "clean":
        confusion_matrix_dir = "clean_confusions_last"
    elif dataset_type == "original":
        confusion_matrix_dir = "original_confusions_last"
    else:
        raise ValueError("Invalid dataset_type. Must be 'original' or 'clean'.")

    # Build the file name based on parameters
    confusion_matrix_file = f"confusion_matrix_class_{class_label}_{dataset_type}_{data_type}.png"

    # Combine directory and file name to get the full path
    confusion_matrix_path = os.path.join(confusion_matrix_dir, confusion_matrix_file)

    return confusion_matrix_path

##########################******* END CODE BLOCK 2 ********#################################



##########################******* START CODE BLOCK 3********#################################
def compute_LRP(image, T, dataset_type="original"):
    """ Compute Layer-wise Relevance Propagation (LRP) for a given image and target label. """

    # Ensure the image is float type
    image = image.astype(np.float32)
    
    # Flatten the image
    X = image.reshape(1, 28 * 28)
   
    # Load parameters
    if dataset_type == "clean":
        W, B = utils.clean_loadparams()  # Load weights and biases for clean dataset
    else:
        W, B = utils.loadparams()  # Load weights and biases for original dataset
    L = len(W)
    
    # Forward pass
    A = [X] + [None] * L
    for l in range(L):
        A[l + 1] = np.maximum(0, A[l].dot(W[l]) + B[l])

    # Check if the score of the new label is positive or negative
    new_label_score = A[L][0, T]

    if new_label_score > 0:
        # Original relevance initialization
        R = [None] * L + [A[L] * (T == np.arange(10))]
        
    else:
        
        R = [None] * L + [
            np.maximum(A[L], 0) * (T == np.arange(10)) -
            np.minimum(A[L], 0) * (T == np.arange(10)) +
            1e-6 * (T == np.arange(10))  # Add baseline relevance for the target label
        ]
        
    # Backward pass
    def rho(w, l): return w + [None, 0.1, 0.0, 0.0][l] * np.maximum(0, w)
    def incr(z, l): return z + [None, 0.0, 0.1, 0.0][l] * (z**2).mean()**0.5 + 1e-9

    for l in range(1, L)[::-1]:
        w = rho(W[l], l)
        b = rho(B[l], l)
        z = A[l].dot(w) + b + 1e-9
        s = R[l + 1] / z
        c = s.dot(w.T)
        R[l] = A[l] * c
       
    # First layer backward pass
    w = W[0]
    wp = np.maximum(0, w)
    wm = np.minimum(0, w)
    lb = A[0] * 0 - 1
    hb = A[0] * 0 + 1

    z = A[0].dot(w) - lb.dot(wp) - hb.dot(wm) + 1e-9
    
    s = R[1] / z
    
    c, cp, cm = s.dot(w.T), s.dot(wp.T), s.dot(wm.T)
    
    R[0] = A[0] * c - lb * cp - hb * cm
    
    return R[0].reshape(28, 28), X.reshape(28, 28)

def LRP_heatmap_with_overlay(R, X, sx, sy, interpolation='bilinear', zoom_factor=1, ax=None):
    """ Plot the LRP heatmap overlayed on the input image. """   
    alpha_overlay=0.7


# Process the background (X) to make the digits white and background gray
    X = X / X.max()  # Normalize X to [0, 1]
    X = 1 - X  # Invert: background becomes dark, digit becomes light
    X = X * 0.5 + 0.5  # Scale background to gray (0.5) and digit to white (1.0)

    # Normalize the LRP heatmap for better visualization
    R = R / np.max(np.abs(R))  # Scale R to [-1, 1]

    # Compute brightness scaling factor
    b = 5 * ((np.abs(R)**3.0).mean()**(1.0/3))  # Adjust factor for brightness

    # Create the custom colormap for the LRP heatmap
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 1  # Keep full intensity
    my_cmap = ListedColormap(my_cmap)

    # Upscale the LRP heatmap to match the target resolution
    if zoom_factor > 1:
        R = scipy.ndimage.zoom(R, zoom=zoom_factor, order=3)  # Cubic interpolation

    # Create the figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(sx, sy))
    else:
        fig = None

    # Remove axis ticks and add padding for better display
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Plot the processed digit image (X) as the background
    ax.imshow(X, cmap='gray', interpolation=interpolation, extent=(0, R.shape[1], 0, R.shape[0]), alpha=1)

    # Overlay the LRP heatmap with transparency
    ax.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, alpha=alpha_overlay, interpolation=interpolation, 
              extent=(0, R.shape[1], 0, R.shape[0]))
    # ax.imshow(X, cmap='gray', interpolation=interpolation, extent=(0, R.shape[1], 0, R.shape[0]), alpha=1.0)

    # If a new figure was created, return it for external handling
    if fig is not None:
        return fig

def compute_class_scores(image, new_label,dataset_type="original"):
    """ Compute the raw scores for all classes using backward relevance propagation. """
   

    # Ensure the image is float type
    image = image.astype(np.float32)

    # Flatten the image
    X = image.reshape(1, 28 * 28)

    # Load parameters
    if dataset_type == "clean":
        W, B = utils.clean_loadparams()  # Load weights and biases for clean dataset
    else:
        W, B = utils.loadparams()  # Load weights and biases for original dataset
    L = len(W)

    # Forward pass
    A = [X] + [None] * L
    for l in range(L - 1):  # Apply ReLU activation only to hidden layers
        A[l + 1] = np.maximum(0, A[l].dot(W[l]) + B[l])

    # Final layer (output layer) without ReLU
    A[L] = A[L - 1].dot(W[L - 1]) + B[L - 1]

    # Modify scores based on the new label
    modified_scores = A[L][0]  # Shape: (10,)

    return modified_scores, new_label


def get_sorted_scores(scores):
    """ Sort the scores in descending order and return the indices and values. """  
    sorted_indices = np.argsort(scores)[::-1]  # Indices of scores sorted in descending order
    sorted_scores = [(i, scores[i]) for i in sorted_indices]
    return sorted_scores
##########################******* START CODE BLOCK 3********#################################


##########################******* START CODE BLOCK 4********#################################

# Function to rotate image
def rotate_image(image, angle):
    from scipy.ndimage import rotate
    # Apply rotation without suppressing minor changes
    return rotate(image, angle, reshape=False)


def predict_new_image(noisy_image,dataset_type="original"):
    """ Predict the label for a given image using the trained model. """

    # Load model parameters
    if dataset_type == "clean":
        W, B = utils.clean_loadparams()  # Load weights and biases for clean dataset
    else:
        W, B = utils.loadparams()  # Load weights and biases for original dataset
    L = len(W)

    # Flatten the image to a vector
    activations = noisy_image.flatten()

    # Perform forward pass through the layers
    for w, b in zip(W, B):
        activations = np.maximum(0, np.dot(activations, w) + b)

    # Predict the label based on the highest activation
    predicted_label = np.argmax(activations)

    return predicted_label


def center_image(image):
    """ Center the input image using the center of mass. """
    cy, cx = center_of_mass(image)  # Find the center of mass
    shift_y, shift_x = np.array(image.shape) // 2 - np.array([cy, cx])
    centered_image = shift(image, shift=(shift_y, shift_x), mode="constant", cval=0)
    return centered_image

# Convert MNIST image to RGB format
def convert_to_rgb(image_array):
    """ Convert a grayscale image to RGB format. """
    pil_image = Image.fromarray((image_array.squeeze() * 255).astype("uint8"), mode="L")
    return pil_image.convert("RGB")


# Convert image to base64 for clickable_images
def convert_to_base64(image):
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

def resize_to_28x28(img):
    st.write("### Debug: Input Image Shape", img.shape)
    img_h, img_w = img.shape
    dim_size_max = max(img.shape)

    # Adjust aspect ratio and resize to fit within 26x26
    if dim_size_max == img_w:
        im_h = (26 * img_h) // img_w
        tmp_img = cv2.resize(img, (26, im_h), interpolation=cv2.INTER_CUBIC)  # Changed interpolation to cv2.INTER_LINEAR
        st.write("### Debug: Temporary Image (Aspect Ratio Adjusted) Shape", tmp_img.shape)
    else:
        im_w = (26 * img_w) // img_h
        tmp_img = cv2.resize(img, (im_w, 26), interpolation=cv2.INTER_CUBIC)  # Changed interpolation to cv2.INTER_LINEAR
        st.write("### Debug: Temporary Image (Aspect Ratio Adjusted) Shape", tmp_img.shape)

    # Create a blank 28x28 image and center the resized image within it
    out_img = np.zeros((28, 28), dtype=np.float32)  # Changed dtype to np.float32

    nb_h, nb_w = out_img.shape
    na_h, na_w = tmp_img.shape
    y_min = (nb_w // 2) - (na_w // 2)
    y_max = y_min + na_w
    x_min = (nb_h // 2) - (na_h // 2)
    x_max = x_min + na_h

    # Copy the resized image into the center of the blank 28x28 image
    out_img[x_min:x_max, y_min:y_max] = tmp_img

    # Normalize the image to [0, 1]


    st.write("### Debug: Final Resized Image Shape", out_img.shape)
    st.write("### Debug: Final Resized Image Pixel Range", out_img.min(), out_img.max())

    return out_img

##########################******* END CODE BLOCK 4 ********#################################


##########################******* START CODE BLOCK 5 ********#################################
def compute_difference_metrics(R_original, R_processed, threshold=1e-8):
    """ Compute pixel-wise difference,cosine_similarity,SSIM and MSE between two heatmaps. """
    if R_original.shape != R_processed.shape:
        raise ValueError("The input heatmaps must have the same shape.")
    
    # Compute pixel-wise difference
    difference = R_processed - R_original
    difference[np.abs(difference) < threshold] = 0  # Ignore small differences

    # Flatten heatmaps for similarity metrics
    flat_original = R_original.flatten()
    flat_processed = R_processed.flatten()

    # Compute cosine similarity
    cosine_sim = cosine_similarity(flat_original.reshape(1, -1), flat_processed.reshape(1, -1))[0, 0]

    # Compute Structural Similarity Index (SSIM)
    ssim_score = ssim(R_original, R_processed, data_range=R_processed.max() - R_processed.min())

    # Compute Mean Squared Error (MSE)
    mse = np.mean((R_original - R_processed) ** 2)

    # Compile results
    metrics = {
        "difference_heatmap": difference,
        "cosine_similarity": cosine_sim,
        "ssim":ssim_score,
        "mse": mse,
    }

    return metrics

def plot_difference_heatmap(difference_heatmap, ax=None):
    """ Plot the difference heatmap between two relevance maps. """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    
    # Plot the heatmap
    heatmap = ax.imshow(
        difference_heatmap,
        cmap="RdBu",  # Red for positive, Blue for negative differences
        interpolation="nearest",
        aspect="equal"
    )
    # Add colorbar
    plt.colorbar(heatmap, ax=ax, orientation="vertical", shrink=0.8, pad=0.05)
    ax.axis("off")
##########################******* END CODE BLOCK 5 ********#################################

