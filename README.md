<<<<<<< HEAD
# XAI_Interface
=======
# Cognitive XAI Interface - Robustness Evaluation

## Project Overview
This project provides an interactive interface for evaluating the robustness of an AI model trained on the MNIST dataset. The primary research question is:

**"How robust is the model in terms of classification and response to adversarial perturbations for both clean and original MNIST datasets?"**

To investigate this, the interface enables users to:
- Observe misclassified examples or randomly selected images from both the original and cleaned MNIST datasets (cleaned using Cleanlab).
- Apply adversarial perturbations by adding noise or rotating images.
- Draw custom digits for model evaluation.
- Generate Layer-wise Relevance Propagation (LRP) heatmaps and score bar graphs.
- Compare LRP heatmaps and bar graphs for original and perturbed images.
- Analyze the pixel-wise difference between LRP results of the original and perturbed images.
- View precision, recall, and F1-score for each predicted class.
- Examine confusion matrices and the percentage of false negatives and false positives.
- Compare robustness using misclassification confidence value distributions and confidence trends under increasing attack intensity.

This project utilizes:
- **Layer-wise Relevance Propagation (LRP):** Implemented following the tutorial available at [LRP Tutorial](https://git.tu-berlin.de/gmontavon/lrp-tutorial).
- **Data Cleaning:** The dataset is cleaned using Cleanlab, based on the methodology described in the [Cleanlab Image Cleaning Tutorial](https://docs.cleanlab.ai/stable/tutorials/datalab/image.html).

---

## Installation Guide
### Step 1: Install Required Libraries
To run the project, install the necessary Python libraries using the following commands:

#### Core Libraries for Data Science and Machine Learning
```bash
pip install numpy matplotlib scipy scikit-learn seaborn
```

#### Deep Learning and Image Processing Libraries
```bash
pip install torch torchvision opencv-python scikit-image
```

#### Web and Interactive Visualization Libraries
```bash
pip install streamlit st-clickable-images streamlit-drawable-canvas streamlit-plotly-events plotly
```

#### Image Handling Library
```bash
pip install pillow
```

#### Other Required Libraries
```bash
pip install pickle5
```
*(Note: pickle is built into Python, but pickle5 may be required for some versions.)*

---

### Step 2: Download MNIST Datasets
Download both the original and cleaned MNIST datasets from the following links:

- **Original MNIST Dataset:** [Download Here](https://www.dropbox.com/scl/fo/6zweanb0qfd1hztbpftu2/AP3F4BCI9USxuAl7u6ObizQ?rlkey=bzdjw8bf15f27lxcnwiugny14&st=8ugnfxf2&dl=0)
- **Cleaned MNIST Dataset:** [Download Here](https://www.dropbox.com/scl/fo/cnscogdpmt4joipuxxs9l/ALJNYG25HDIjAuGmxCRadFk?rlkey=el5k6r8h34ytibanzk2u2w1v0&st=7natyog3&dl=0)

Ensure the datasets are stored in the appropriate directories before running the interface.

---

### Step 3: Running the Interface
The Streamlit-based interface is implemented in `Interface.py`. To launch the application, use the following command:

```bash
streamlit run Interface.py
```

This will start the web-based interface, allowing you to interactively evaluate model robustness through various functionalities.

---

## Usage Instructions
Once the interface is running:
1. Choose between **misclassified examples** or **random images** from the dataset.
2. Select an image from the **original or cleaned dataset** (train/test split available).
3. Apply transformations such as **adding noise or rotating the image**.
4. Observe the **LRP heatmap and score bar graph** for the selected image.
5. Compare the **LRP and score bar graph** of the original vs. perturbed image.
6. Analyze **recall, precision, and F1-score** for the predicted class.
7. Examine the **confusion matrix** and **false negative/false positive rates**.
8. View **statistical visualizations** comparing original and cleaned dataset performance.
9. Observe how model confidence changes as **attack intensity increases**.

---

## Code Explanation Blocks
If you want to find the function(s) used in each part of interface open them code_map.jpg, each code block can be found in Helper_Functions.py like following : 

######******* START CODE BLOCK 1********######


.
.
.


#########******* END CODE BLOCK 1********#####

---

## High Process Functions 
There are functions used in this project ranging from training the model to finding overall confidence value of all misclassified examples WRT adversarial attacks which we already run them.
These Functions are located in High_process_functions.py and excuted in High_process_functions_execution.py. 

---

---


>>>>>>> master
