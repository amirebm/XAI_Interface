import streamlit as st
import matplotlib.pyplot as plt
from st_clickable_images import clickable_images
from scipy.ndimage import gaussian_filter, zoom
from streamlit_plotly_events import plotly_events  # Add this library to handle Plotly click events
import plotly.graph_objects as go
from streamlit.runtime.scriptrunner import RerunException
from streamlit.runtime.scriptrunner import get_script_run_ctx
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import os,cv2
import streamlit_shadcn_ui as ui
import os
from streamlit_modal import Modal
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from torchvision import datasets, transforms

st.set_page_config(layout="wide")
from PIL import Image
import utils_saves
from High_process_functions import (
   

    calculate_class_metrics_clean,
    calculate_class_metrics_original,
    visualize_misclassified_confidence_distribution,
    load_model_parameters,
    evaluate_model_confidence_drop_original,
    evaluate_model_confidence_drop_clean,
    pgd_attack,
    save_plot,
    check_data_sizes,
    evaluate_accuracy,
    evaluate_confidence,
    count_clean_train_data,
    evaluate_saved_models_with_metrics,
    visualize_misclassified_confidence_distribution,
    visualize_misclassified_confidence_distribution_by_class,
    calculate_overall_metrics,
    calculate_confusion_matrix_per_class_clean,
    calculate_confusion_matrix_per_class_original,
    train_model_last,
    set_model_params,

)


################# misclassified_confidence_distribution for model and each class ###################
visualize_misclassified_confidence_distribution()
print("misclassified_confidence_distribution for model finished")
visualize_misclassified_confidence_distribution_by_class()
print("misclassified_confidence_distribution for each class finished")

#################MISCLASSIFIED EXAMPLES###################

##################SAVE CONFUSION MATRIX ##################

calculate_overall_metrics(data_type="test",  dataset_size=10000,dataset_type="original")
calculate_overall_metrics(data_type="train", dataset_size=60000,dataset_type="original")
print("calculate_overall_metrics for original dataset finished")

calculate_overall_metrics(data_type="test",  dataset_size=10000,dataset_type="clean")
calculate_overall_metrics(data_type="train", dataset_size=56593,dataset_type="clean")
print("calculate_overall_metrics for clean dataset finished")
##################SAVE CONFUSION MATRIX ##################


################TRAIN MODEL######################
train_model_last(data_type='original', epochs=50, batch_size=64, learning_rate=0.001, save_dir="new_params_last_27_01")
train_model_last(data_type='clean', epochs=50, batch_size=64, learning_rate=0.001, save_dir="clean_new_params_last_27_01")


################TRAIN MODEL######################


##########################evaluating the model###############
# Generate the report for saved models


evaluate_saved_models_with_metrics(
    original_params_loader=utils_saves.loadparams,
    clean_params_loader=utils_saves.clean_loadparams,
    batch_size=64
)

##########################evaluating the model###############




###################### ****Confidence value drop WRT PGD attacks**** #####################



# Define parameters
attack_strengths = np.linspace(0, 0.5, 50)

model_original = utils_saves.SimpleDeepRectifierNet()
model_clean = utils_saves.SimpleDeepRectifierNet()

# Load weights and biases
W_original, B_original = utils_saves.loadparams()
W_clean, B_clean = utils_saves.clean_loadparams()



# Set parameters for both models
set_model_params(model_original, W_original, B_original)
set_model_params(model_clean, W_clean, B_clean)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_original.to(device)
model_clean.to(device)

# Ensure models are in evaluation mode
model_original.eval()
model_clean.eval()

# Evaluate both models
results_original = evaluate_model_confidence_drop_original(
    model_original, utils_saves.loaddata, utils_saves.loaddata_clean, pgd_attack, attack_strengths
)
results_clean = evaluate_model_confidence_drop_clean(
    model_clean, utils_saves.loaddata, utils_saves.loaddata_clean, pgd_attack, attack_strengths
)

# Save the plot
save_plot(attack_strengths, {
    "original": results_original["original"],
    "clean": results_clean["clean"]
}, output_path="confidence_WRT_attacks")

###################### ****Confidence value drop WRT PGD attacks**** #####################

######################confidence wrt attacks#####################


##############comparing model performance on original and  clean data##########
#Load models

clean_train_path = "clean_data_last/train_clean.pkl"

# Count and print the number of samples
num_clean_train_samples = count_clean_train_data(clean_train_path)
print(f"Number of clean training samples: {num_clean_train_samples}")


model_original = utils_saves.SimpleDeepRectifierNet()
model_clean = utils_saves.SimpleDeepRectifierNet()

# Load weights and biases
load_model_parameters(model_original, utils_saves.loadparams()[0], utils_saves.loadparams()[1])
load_model_parameters(model_clean, utils_saves.clean_loadparams()[0], utils_saves.clean_loadparams()[1])

# Check data sizes
check_data_sizes()

# Evaluate accuracy
num_original_samples = 10000  # Adjust based on Step 1 results
num_clean_samples = 10000      # Adjust based on Step 1 results

evaluate_accuracy(model_original, utils_saves.loaddata, num_original_samples, "Original")
evaluate_accuracy(model_clean, utils_saves.loaddata_clean, num_clean_samples, "Clean")


evaluate_confidence(model_original, utils_saves.loaddata, num_original_samples, "Original")
evaluate_confidence(model_clean, utils_saves.loaddata_clean, num_clean_samples, "Clean")

##############comparing model performance on original and  clean data##########3



######################F1-SCORE AND CONFUSION MATRIX####################
calculate_class_metrics_clean(data_type="test", num_classes=10,dataset_size=10000)
print("calculate_class_metrics_clean for test finished")
calculate_class_metrics_clean(data_type="train", num_classes=10,dataset_size=56593)
print("calculate_class_metrics_clean for train finished")


calculate_class_metrics_original(data_type="test", num_classes=10, dataset_size=10000,dataset_type="original")
print("original class metrics for test finished")
calculate_class_metrics_original(data_type="train", num_classes=10, dataset_size=60000,dataset_type="original")
print("original class metrics for train finished")

###############CONFUSION MATRIX Calculation Per Class##############################
calculate_confusion_matrix_per_class_original(data_type="test", dataset_size=10000,dataset_type="original")
calculate_confusion_matrix_per_class_original(data_type="train", dataset_size=60000,dataset_type="original")
print("original finished")

calculate_confusion_matrix_per_class_clean(data_type="test",dataset_type="clean")
calculate_confusion_matrix_per_class_clean(data_type="train",dataset_type="clean")
print("clean  finished")

###############CONFUSION MATRIX Calculation Per Class##############################3

