import streamlit as st
import matplotlib.pyplot as plt
from st_clickable_images import clickable_images

from streamlit_plotly_events import plotly_events  # Add this library to handle Plotly click events
import plotly.graph_objects as go
from streamlit.runtime.scriptrunner import RerunException
from streamlit.runtime.scriptrunner import get_script_run_ctx
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os,cv2
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


st.set_page_config(layout="wide")
from PIL import Image

from Helper_Functions import (
    convert_to_rgb,
    convert_to_base64,
    LRP_heatmap_with_overlay,
    reload_random_images,
    reload_misclassified_images,
    compute_LRP,
    rotate_image,
    predict_new_image,
    compute_difference_metrics,
    plot_difference_heatmap,
    compute_class_scores,
    get_sorted_scores,
    center_image,
    get_class_metrics,
    get_confusion_matrix_path,
    get_confidence_distribution_path,
    get_model_metrics,
)

def initialize_session_state():
    for key, value in {
        "bar_clicked": False,
        "drawn_bar_clicked": False,
        "true_label": False,
        "processed_bar_clicked": False,
        "image_data": None,
        "selected_image_index": None,
        "current_display": "misclassified",
        "image_selected": False,
        "active_image_state": None,
        "rotation_angle": 0,
        "noisy_image": None,
        "rotated_image": None,
        "user_interacted_with_settings": False,
        "user_interacted": False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value
initialize_session_state()


print("loading")

# Initialize session state variables
if "bar_clicked" not in st.session_state:
    st.session_state["bar_clicked"] = False  
if "drawn_bar_clicked" not in st.session_state:
    st.session_state["drawn_bar_clicked"] = False  
if "true_label" not in st.session_state:
    st.session_state["true_label"] = False  
if "processed_bar_clicked" not in st.session_state:
    st.session_state["processed_bar_clicked"] = False 
if "image_data" not in st.session_state:
    st.session_state["image_data"] = None
if "selected_image_index" not in st.session_state:
    st.session_state["selected_image_index"] = None
if "current_display" not in st.session_state:
    st.session_state["current_display"] = "misclassified"  
if "image_selected" not in st.session_state:
    st.session_state["image_selected"] = False  
if "active_image_state" not in st.session_state:
    st.session_state["active_image_state"] = None  
if "rotation_angle" not in st.session_state:
    st.session_state["rotation_angle"] = 0
if "noise_level" not in st.session_state:
    st.session_state["noise_level"] = 0
if "noisy_image" not in st.session_state:
    st.session_state["noisy_image"] = None
if "rotated_image" not in st.session_state:
    st.session_state["rotated_image"] = None
if "user_interacted_with_settings" not in st.session_state:
    st.session_state["user_interacted_with_settings"] = False
if "user_interacted" not in st.session_state:
    st.session_state["user_interacted"] = False
if "dataset_type" not in st.session_state:
    st.session_state["dataset_type"] = "original"
if "data_type" not in st.session_state:
    st.session_state["data_type"] = "test"
if "cleared" not in st.session_state:
    st.session_state["cleared"] = False 

# Check if any of the radio buttons have changed
if ("last_dataset_type" not in st.session_state or 
    st.session_state["dataset_type"] != st.session_state.get("last_dataset_type")) or \
   ("last_data_type" not in st.session_state or 
    st.session_state["data_type"] != st.session_state.get("last_data_type")):

    # Update the session states for the new choices
    st.session_state["last_dataset_type"] = st.session_state["dataset_type"]
    st.session_state["last_data_type"] = st.session_state["data_type"]

    # Reload images based on the updated parameters
    random_images_data = reload_random_images(
        data_type=st.session_state["data_type"], 
        dataset=st.session_state["dataset_type"]
    )
    
    misclassified_examples = reload_misclassified_images(
        data_type=st.session_state["data_type"], 
        dataset=st.session_state["dataset_type"]
    )

    # Update the session state with the newly loaded data
    st.session_state["image_data"] = {
        "random": random_images_data,
        "misclassified": misclassified_examples,
    }

tab1, tab2 = st.tabs(["Main Interface", "Statistics"])

with tab1:
    # Sidebar for options
    with st.sidebar:
        st.header("Dataset Settings")
    # Radio button to choose between original and clean data

        st.radio(
            "Choose Dataset Type",
            ["original", "clean"],
            index=0 ,
            key="dataset_type",
        )
        # Radio button to choose between test and train data

        st.radio(
            "Choose Data Type",
            ["test", "train"],
            index=0 ,
            key="data_type",
        )
        st.radio(
            "Choose Grid Content",
            ["misclassified", "random"],
            index=0,
            key="current_display",
        )

        
    with st.sidebar:
        st.subheader("Noise Settings")
        # Noise slider within the sidebar
        noise_level = st.slider("Noise Level", 0, 100, 0, key="noise_level")
        if noise_level > 0:
            st.session_state["user_interacted_with_settings"] = True  # Mark interaction
        
        st.subheader("Rotation Settings")
        rotate_left = st.button("🔄 Rotate Left", key="rotate_left")
        rotate_right = st.button("🔄 Rotate Right", key="rotate_right")
        if rotate_left or rotate_right:
            st.session_state["user_interacted_with_settings"] = True  # Mark interaction


        st.subheader("Draw Yourself")
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0.0)",  # Fully transparent fill
            stroke_width=15,  # Fixed stroke width
            stroke_color="#FFFFFF",  # Drawing color set to white
            background_color="#000000",  # Black background
            height=200,  # Large visible canvas for drawing
            width=200,   # Large visible canvas for drawing
            drawing_mode="freedraw",  # Default to "freedraw" mode
            key="canvas",
        )
        import tempfile
    # Check if the display type has changed
    if "last_display" not in st.session_state:
        st.session_state["last_display"] = st.session_state["current_display"]

    if st.session_state["current_display"] != st.session_state["last_display"]:
        # Reset variables controlling plots, displayed images, noise, and rotation
        for key in [
            "selected_image",
            "selected_image_index",
            "true_label",
            "rotation_angle",
            "noisy_image",
            "rotated_image",
            "user_interacted",
            "bar_clicked",
            "processed_bar_clicked",
            "active_image_state",
            # "noise_level",  # Reset noise slider value
        ]:
            if key in st.session_state:
                st.session_state[key] = None  # Reset to default value or clear it

        # Reset specific session states
        st.session_state["image_selected"] = False  # Reset image selection state
        st.session_state["rotation_angle"] = 0  # Reset rotation angle
        # st.session_state["noise_level"] = 0  # Reset noise slider to 0
        st.session_state["last_display"] = st.session_state["current_display"]  # Update the last display type

    
    # Fetch the appropriate data based on user preference
    if st.session_state["current_display"] == "random":
        grid_title = "Random Images"
        images = st.session_state["image_data"]["random"]["images"]
        titles = [
            f"True: {true} | Pred: {pred}"
            for true, pred in zip(
                st.session_state["image_data"]["random"]["true_labels"],
                st.session_state["image_data"]["random"]["predictions"],
            )
        ]
    else:
        grid_title = "Misclassified Images"
        images = [example["image"] for example in st.session_state["image_data"]["misclassified"]]
        titles = [
            f"True: {example['true_label']} | Pred: {example['predicted_label']}"
            for example in st.session_state["image_data"]["misclassified"]
        ]

    # Display grid title
    st.markdown(f"<h2 style='text-align: center;'>{grid_title}</h2>", unsafe_allow_html=True)

    images_rgb = [convert_to_rgb(img) for img in images]
    base64_images = [f"data:image/png;base64,{convert_to_base64(img)}" for img in images_rgb]


    # Render image grid
    clicked_index = clickable_images(
        paths=base64_images,
        titles=titles,
        div_style={
            "display": "flex",
            "justify-content": "center",
            "flex-wrap": "wrap",
            "gap": "5px",
        },
        img_style={
            "margin": "5px",
            "height": "100px",  # Increased from 70px to 84px (20% increase)
            "border-radius": "5px",
            "cursor": "pointer",
        },
    )
    refresh_button_container = st.container()
    with refresh_button_container:
        if st.button("🔄 Refresh"):
            if st.session_state["current_display"] == "random":
                st.session_state["image_data"]["random"] = reload_random_images( data_type=st.session_state["data_type"], 
            dataset=st.session_state["dataset_type"])
            else:
                st.session_state["image_data"]["misclassified"] = reload_misclassified_images( data_type=st.session_state["data_type"], 
            dataset=st.session_state["dataset_type"])
            st.session_state["selected_image_index"] = None
            st.session_state["image_selected"] = False
            st.session_state.pop("selected_image", None)
                    
    print("image",clicked_index)

    # Display LRP heatmap and original image when an image is clicked
    if clicked_index > -1:  
        st.session_state["selected_image_index"] = clicked_index
        st.session_state["image_selected"] = True
        st.session_state["bar_clicked"] = False  # Reset when a new image is clicked
        st.session_state["processed_bar_clicked"] = False
        if st.session_state["current_display"] == "random":
            original_index = int(st.session_state["image_data"]["random"]["indices"][clicked_index])
            selected_image = st.session_state["image_data"]["random"]["images"][clicked_index]
            true_label = st.session_state["image_data"]["random"]["true_labels"][clicked_index]
            pred_label = st.session_state["image_data"]["random"]["predictions"][clicked_index]
        else:
            original_index = int(st.session_state["image_data"]["misclassified"][clicked_index]["id"])
            selected_image = st.session_state["image_data"]["misclassified"][clicked_index]["image"]
            true_label = st.session_state["image_data"]["misclassified"][clicked_index]["true_label"]
            pred_label = st.session_state["image_data"]["misclassified"][clicked_index]["predicted_label"]

        # Store the selected image in session state for noise settings
        st.session_state["selected_image"] = selected_image
        
        # Ensure `true_label` is initialized
        st.session_state["true_label"] = true_label
        # Handle rotation buttons
        if rotate_right:
            st.session_state["rotation_angle"] = st.session_state.get("rotation_angle", 0) - 15
        elif rotate_left:
            st.session_state["rotation_angle"] = st.session_state.get("rotation_angle", 0) + 15

        try:
            # Compute LRP and get scores
            if st.session_state["dataset_type"] == "original":
                    R, X = compute_LRP(selected_image, pred_label)
                    final_scores, target_label = compute_class_scores(selected_image, pred_label)
            elif st.session_state["dataset_type"] == "clean":
                    R, X = compute_LRP(selected_image, pred_label,dataset_type="clean")
                    final_scores, target_label = compute_class_scores(selected_image, pred_label,dataset_type="clean")
            
            sorted_scores = get_sorted_scores(final_scores)
        
            if st.session_state["dataset_type"] == "original" and st.session_state["data_type"] == "test":
                class_metrics = get_class_metrics(true_label, data_type="test", dataset_type="original")
            elif st.session_state["dataset_type"] == "original" and st.session_state["data_type"] == "train":
                class_metrics = get_class_metrics(true_label, data_type="train", dataset_type="original")
            elif st.session_state["dataset_type"] == "clean" and st.session_state["data_type"] == "test":
                class_metrics = get_class_metrics(true_label, data_type="test", dataset_type="clean")
            elif st.session_state["dataset_type"] == "clean" and st.session_state["data_type"] == "train":
                class_metrics = get_class_metrics(true_label, data_type="train", dataset_type="clean")
            else:
                st.error("Invalid dataset_type or data_type in session state.")

            f1_score = class_metrics["f1_score"]
            precision = class_metrics["precision"]
            recall = class_metrics["recall"]
            # Directory containing confusion matrix images
            confusion_matrix_dir = "clean_class_metrics_last"
                # Display metrics at the top
            col1, col2,col3 = st.columns([2,1,1])  # Adjust column widths as needed

            with col1:
                st.markdown(
                    f"""
                    <h3 style='text-align: center; font-size: 40px; color: purple;'>Label: {true_label}</h3> <!-- Twice as big and purple -->
                    <p style='text-align: center; color: navy; font-size: 32px;'>
                    <b>F1-Score:</b> {f1_score:.4f} | 
                    <b>Precision:</b> {precision:.4f} | 
                    <b>Recall:</b> {recall:.4f}
                    </p>
                    """,
                    unsafe_allow_html=True,
                ) 
            # Define the dialog function
            @st.dialog("Confusion Matrix Analysis", width="large")
            def show_confusion_matrix():
                if st.session_state["dataset_type"] == "original" and st.session_state["data_type"] == "test":
                    confusion_matrix_path = get_confusion_matrix_path(true_label, data_type="test", dataset_type="original")
                elif st.session_state["dataset_type"] == "original" and st.session_state["data_type"] == "train":
                    confusion_matrix_path = get_confusion_matrix_path(true_label, data_type="train", dataset_type="original")
                elif st.session_state["dataset_type"] == "clean" and st.session_state["data_type"] == "test":
                    confusion_matrix_path = get_confusion_matrix_path(true_label, data_type="test", dataset_type="clean")
                elif st.session_state["dataset_type"] == "clean" and st.session_state["data_type"] == "train":
                    confusion_matrix_path = get_confusion_matrix_path(true_label, data_type="train", dataset_type="clean")
                else:
                    st.error("Invalid dataset_type or data_type in session state.")

                        # Check if the file exists
                if os.path.exists(confusion_matrix_path):
                    # Load the confusion matrix image
                    high_res_image = plt.imread(confusion_matrix_path)

                    # Create a figure for displaying the confusion matrix
                    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)  # Adjust size as needed
                    ax.imshow(high_res_image, cmap="gray")
                    ax.axis("off")  # Hide axes for a clean display

                    # Render the confusion matrix inside the dialog
                    st.pyplot(fig)

                    # Close button to dismiss the dialog
                
                else:
                    st.error(f"Confusion matrix image not found for class {true_label}.")
            

            @st.dialog("Confidence Value Analysis", width="large")
            def show_confidence_value():
                if st.session_state["dataset_type"] == "original" and st.session_state["data_type"] == "test":
                    confidence_value_path = get_confidence_distribution_path(true_label, data_type="test")
                elif st.session_state["dataset_type"] == "original" and st.session_state["data_type"] == "train":
                    confidence_value_path = get_confidence_distribution_path(true_label, data_type="train")
                elif st.session_state["dataset_type"] == "clean" and st.session_state["data_type"] == "test":
                    confidence_value_path = get_confidence_distribution_path(true_label, data_type="test")
                elif st.session_state["dataset_type"] == "clean" and st.session_state["data_type"] == "train":
                    confidence_value_path = get_confidence_distribution_path(true_label, data_type="train")
                else:
                    st.error("Invalid dataset_type or data_type in session state.")

                # Check if the file exists
                if os.path.exists(confidence_value_path):
                    # Load the confusion matrix image
                    high_res_image = plt.imread(confidence_value_path)

                    # Create a figure for displaying the confusion matrix
                    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)  # Adjust size as needed
                    ax.imshow(high_res_image, cmap="gray")
                    ax.axis("off")  # Hide axes for a clean display

                    # Render the confusion matrix inside the dialog
                    st.pyplot(fig)

                    # Close button to dismiss the dialog
                
                else:
                    st.error(f"Confidence Value image not found for class {true_label}.")


            # Button to trigger the dialog
            with col3:
                if st.button("Class Confusion Matrix"):
                    # Call the dialog function to show the popup
                    show_confusion_matrix()
                
                if st.button("Class Confidence Value"):
                    # Call the dialog function to show the popup
                    show_confidence_value() 
            # Display metrics at the top with a gray line separator
            st.markdown(
                f"""
                <div style='border-bottom: 1px solid gray; margin-bottom: 10px;'></div> <!-- Add gray line -->
                
                """,
                unsafe_allow_html=True,
            )
            with col2:
                if st.session_state["dataset_type"] == "original" and st.session_state["data_type"] == "test":
                    metrics = get_model_metrics(data_type="test", dataset_type="original")
                elif st.session_state["dataset_type"] == "original" and st.session_state["data_type"] == "train":
                    metrics = get_model_metrics(data_type="train", dataset_type="original")
                elif st.session_state["dataset_type"] == "clean" and st.session_state["data_type"] == "test":
                    metrics = get_model_metrics(data_type="test", dataset_type="clean")
                elif st.session_state["dataset_type"] == "clean" and st.session_state["data_type"] == "train":
                    metrics = get_model_metrics(data_type="train", dataset_type="clean") 
                else:
                    st.error("Invalid dataset_type or data_type in session state.")
                
                if "f1_score" in metrics:
                    overall_f1_score = metrics["f1_score"]
                else:
                    print("F1-Score not found in metrics.")
             
                overall_datatype=st.session_state["dataset_type"]  
                overall_dataset=st.session_state["data_type"] 
                st.markdown(
                    f"""
                    <h3 style='text-align: center; font-size: 40px; color: purple;'>Overall</h3> <!-- Twice as big and purple -->
                    <p style='text-align: center; color: navy; font-size: 32px;'>
                     {overall_datatype} | 
                     {overall_dataset} | 
                    <b>F1-Score:</b> {overall_f1_score}
                    </p>
                    """,
                    unsafe_allow_html=True,
                ) 


    # Create a single row layout for all three components
            with st.expander("Original Image and LRP", expanded=True):
            # Keep the rest of the code within this expander
                col1, col2, col3 = st.columns([2, 1, 1])
                if not st.session_state["bar_clicked"]:
                        # Center column: First LRP Heatmap
                        with col3:
                            st.markdown(f"<h4 style='text-align: center;'>Label: {true_label} | Pred: {pred_label}</h4>", unsafe_allow_html=True)
                            # st.write("predicssssssssss",pred_label)
                    
                            # Convert image to PIL format and upscale using LANCZOS
                            pil_image = Image.fromarray((selected_image * 255).astype("uint8"))
                            high_res_image = pil_image.resize((28 * 8, 28 * 8), Image.LANCZOS)  # 8x upscale for higher resolution

                            # Render the original image with consistent size and alignment
                            origin_fig, ax = plt.subplots(figsize=(1, 1), dpi=150)  # Match size to LRP heatmap
                            ax.imshow(high_res_image, cmap="gray")
                            ax.axis("off")
                            st.pyplot(origin_fig)
                            plt.close(origin_fig)
                        with col2:
                            
                            st.markdown(f"<h3 style='text-align: center;'>LRP Heatmap</h3>", unsafe_allow_html=True)
                            origin_fig, ax = plt.subplots(figsize=(1, 1), dpi=150)  # Match size and quality
                            LRP_heatmap_with_overlay(R, X, 9, 0.75, interpolation="bicubic", zoom_factor=2, ax=ax)
                            st.pyplot(origin_fig)
                            plt.close(origin_fig)

                        with col1:
                            st.markdown(f"<h3 style='text-align: center;'>Class Scores WRT  Label {true_label}</h3>", unsafe_allow_html=True)
                            # Prepare data for the bar chart
                            bar_data = {
                                "Class": [str(score[0]) for score in sorted_scores],
                                "Score": [score[1] for score in sorted_scores],
                            }

                            # Create the bar chart with Plotly
                            import plotly.graph_objects as go

                            # Update marker color logic to highlight the target label
                            origin_fig = go.Figure(
                                data=[
                                    go.Bar(
                                        x=bar_data["Score"],
                                        y=bar_data["Class"],
                                        orientation="h",
                                        text=None,  # Remove text inside bars
                                        marker=dict(
                                            color=[
                                                "blue" if int(score[0]) == target_label else ("green" if score[1] > 0 else "red")
                                                for score in sorted_scores
                                            ]
                                        ),
                                    )
                                ]
                            )

                            # Update layout to match your requirements
                            origin_fig.update_layout(
                                hoverlabel=dict(
                                    bgcolor="rgba(0, 0, 0, 1)",  # Black with 80% opacity for background
                                    font=dict(color="white", size=12),  # White hover text with font size 12
                                    namelength=-1,  # Display full class names
                                ),
                                title_font=dict(size=20, color="black"),  # Title font
                                xaxis=dict(
                                    title="Score",
                                    title_font=dict(size=16, color="black"),
                                    tickfont=dict(size=12, color="black"),
                                    zeroline=True,
                                    zerolinecolor="black",
                                ),
                                yaxis=dict(
                                    title="Class",
                                    title_font=dict(size=16, color="black"),
                                    tickfont=dict(size=12, color="black"),
                                ),
                                font=dict(color="black"),
                                plot_bgcolor="white",
                                template="simple_white",
                                height=400,
                                margin=dict(l=10, r=10, t=50, b=10),
                            )

                            # Add cursor styling
                            origin_fig.update_traces(hoverinfo="x+y", hoverlabel=dict(font=dict(size=12)), selector=dict(type="bar"))
                            origin_fig.update_layout(dragmode=False)

                        

                            selected_points = plotly_events(origin_fig, click_event=True)
            
            
            if selected_points:
                    
                    new_label = int(bar_data["Class"][selected_points[0]["pointIndex"]])  # Get the clicked class

                    # Set the session state variable to True when a bar is clicked
                    st.session_state["bar_clicked"] = True
                    st.session_state["selected_label"] = new_label  # Save the selected label to session state
                    # Recompute class scores and LRP heatmap with the new label
                    
                    if st.session_state["dataset_type"] == "original":
                        R, X = compute_LRP(selected_image, new_label)
                        final_scores, target_label = compute_class_scores(selected_image, new_label)
                    elif st.session_state["dataset_type"] == "clean":
                        R, X = compute_LRP(selected_image, new_label,dataset_type="clean")
                        final_scores, target_label = compute_class_scores(selected_image, new_label,dataset_type="clean")

                    sorted_scores_new_label = get_sorted_scores(final_scores)
                    
                    
                # Display LRP heatmap and bar graph only if bar_clicked is True
                    if st.session_state.get("bar_clicked", False) :
                        with st.expander("Click Bar Image and LRP", expanded=True):
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                
                                st.markdown(f"<h3 style='text-align: center;'>Class Scores WRT {target_label} Label </h3>", unsafe_allow_html=True)
                                # Prepare data for the bar chart
                                bar_data = {
                                    "Class": [str(score[0]) for score in sorted_scores_new_label],
                                    "Score": [score[1] for score in sorted_scores_new_label],
                                }

                                # Create the bar chart with Plotly
                                origin_fig = go.Figure(
                                    data=[
                                        go.Bar(
                                            x=bar_data["Score"],
                                            y=bar_data["Class"],
                                            orientation="h",
                                            text=None,  # Remove text inside bars
                                            marker=dict(
                                                color=[
                                                    "blue" if int(score[0]) == target_label else ("green" if score[1] > 0 else "red")
                                                    for score in sorted_scores_new_label
                                                ]
                                            ),
                                        )
                                    ]
                                )

                                # Update layout to match your requirements
                                origin_fig.update_layout(
                                    hoverlabel=dict(
                                        bgcolor="rgba(0, 0, 0, 1)",  # Black with 80% opacity for background
                                        font=dict(color="white", size=12),  # White hover text with font size 12
                                        namelength=-1,  # Display full class names
                                    ),
                                    title_font=dict(size=20, color="black"),  # Title font
                                    xaxis=dict(
                                        title="Score",
                                        title_font=dict(size=16, color="black"),
                                        tickfont=dict(size=12, color="black"),
                                        zeroline=True,
                                        zerolinecolor="black",
                                    ),
                                    yaxis=dict(
                                    title="Class",
                                    title_font=dict(size=16, color="black"),
                                    tickfont=dict(size=12, color="black"),
                                ),
                                font=dict(color="black"),
                                plot_bgcolor="white",
                                template="simple_white",
                                height=400,
                                margin=dict(l=10, r=10, t=50, b=10),
                            )

                                # Add cursor styling
                                origin_fig.update_traces(hoverinfo="x+y", hoverlabel=dict(font=dict(size=12)), selector=dict(type="bar"))
                                origin_fig.update_layout(dragmode=False)

                                # Display the interactive bar chart
                                st.plotly_chart(origin_fig, use_container_width=True)

                            with col2:
                                st.markdown(f"<h3 style='text-align: center;'>LRP Heatmap for Class {st.session_state['selected_label']}</h3>", unsafe_allow_html=True)
                                origin_fig, ax = plt.subplots(figsize=(1, 1), dpi=150)  # Match size and quality
                                LRP_heatmap_with_overlay(R, X, 9, 0.75, interpolation="bicubic", zoom_factor=2, ax=ax)
                                
                                st.pyplot(origin_fig)
                                plt.close(origin_fig)

                            

        except Exception as e:
            st.error(f"Error generating heatmap or class scores: {e}")


    # Handle noisy image display
    # Update the noisy image display section to account for rotation
    # Handle noisy image display and rotation
    if st.session_state.get("image_selected", False):
        # Check if user interacted with noise slider or rotation buttons
        if st.session_state.get("image_selected", False):
        # Safely check or initialize user_interacted in session state
            st.session_state["user_interacted"] = (
                st.session_state.get("noise_level", 0) > 0 or
                st.session_state.get("rotation_angle", 0) != 0
            )
        # Use the true_label from session state, or default to a safe value (e.g., 0)
        true_label = st.session_state.get("true_label", 0)
        # Start with the selected image
        processed_image = st.session_state["selected_image"]

        
        # Apply rotation if rotation angle is set
        rotation_angle = st.session_state.get("rotation_angle", 0)
        if rotation_angle != 0:
            processed_image = rotate_image(processed_image, angle=rotation_angle)

        # Apply noise if noise level is greater than 0
        noise_level = st.session_state.get("noise_level", 0)
        if noise_level > 0:
            processed_image = np.clip(
                processed_image + np.random.normal(0, noise_level / 255.0, processed_image.shape),
                0,
                1,
            )

        # Conditionally display processed (noisy/rotated) image and its LRP only if user interacted
        if st.session_state.get("user_interacted", False):  # Correct

            # Predict the label for the processed (rotated + noisy) image

            if st.session_state["dataset_type"] == "original":
                processed_pred_label = predict_new_image(processed_image,dataset_type="original")

            elif st.session_state["dataset_type"] == "clean":
                processed_pred_label = predict_new_image(processed_image,dataset_type="clean")
                
            with st.expander("Processed Image and Prediction", expanded=True):
                # Display the processed image and prediction
                col1, col2, col3 = st.columns([2, 1, 1])
                # col4, col3 = st.columns(2)

                if st.session_state["dataset_type"] == "original":
                    processed_scores, processed_target_label = compute_class_scores(processed_image, processed_pred_label)

                elif st.session_state["dataset_type"] == "clean":
                    processed_scores, processed_target_label = compute_class_scores(processed_image, processed_pred_label,dataset_type="clean")

                
                processed_sorted_scores = get_sorted_scores(processed_scores)

                
                with col1:
                    st.markdown(f"<h3 style='text-align: center;'>Processed Image LRP WRT Label {processed_target_label}</h3>", unsafe_allow_html=True)

                    # Prepare data for the bar chart
                    processed_bar_data = {
                        "Class": [str(score[0]) for score in processed_sorted_scores],
                        "Score": [score[1] for score in processed_sorted_scores],
                    }

                    # Create the bar chart for processed image
                    processed_fig = go.Figure(
                        data=[
                            go.Bar(
                                x=processed_bar_data["Score"],
                                y=processed_bar_data["Class"],
                                orientation="h",
                                text=None,
                                marker=dict(
                                    color=[
                                        "blue" if int(score[0]) == processed_target_label else ("green" if score[1] > 0 else "red")
                                        for score in processed_sorted_scores
                                    ]
                                ),
                            )
                        ]
                    )
                                # Add cursor styling
                    processed_fig.update_traces(hoverinfo="x+y", hoverlabel=dict(font=dict(size=12)), selector=dict(type='bar'))
                    processed_fig.update_layout(dragmode=False, hoverlabel=dict(namelength=-1))  # Ensure hand cursor
                    processed_fig.update_layout(hoverlabel=dict(font=dict(color="black"))) # make label visible

                    # Update layout to match requirements
                    processed_fig.update_layout(
                    hoverlabel=dict(
                        bgcolor="rgba(0, 0, 0, 1)",  # Black with 80% opacity
                        font=dict(color="white", size=12),  # White text with font size 12
                        namelength=-1,  # Display full class names if needed
                    ),
                    title_font=dict(size=20, color="black"),
                    xaxis=dict(
                        title="Score",
                        title_font=dict(size=16, color="black"),
                        tickfont=dict(size=12, color="black"),
                        zeroline=True,
                        zerolinecolor="black",
                    ),
                    yaxis=dict(
                        title="Class",
                        title_font=dict(size=16, color="black"),
                        tickfont=dict(size=12, color="black"),
                    ),
                    font=dict(color="black"),
                    plot_bgcolor="white",
                    template="simple_white",
                    height=400,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                    # Display the interactive bar chart
                    processed_selected_points = plotly_events(processed_fig, click_event=True)

                    # If the user clicks on the processed bar graph
                    if processed_selected_points:
                            
                    # Set the session state variable to True when a bar is clicked
                        st.session_state["processed_bar_clicked"] = True

                        # Retrieve the clicked class (new_label) from the selected bar
                        processed_new_label = int(processed_bar_data["Class"][selected_points[0]["pointIndex"]])  # Get the clicked class

                        # Recompute class scores and LRP heatmap with the new label
                        if st.session_state["dataset_type"] == "original":
                            R, X = compute_LRP(selected_image, new_label,dataset_type="original")
                            final_scores, target_label = compute_class_scores(selected_image, processed_new_label,dataset_type="original")
                        elif st.session_state["dataset_type"] == "clean":
                            R, X = compute_LRP(selected_image, new_label,dataset_type="clean")
                            final_scores, target_label = compute_class_scores(selected_image, processed_new_label,dataset_type="clean")
                        
                        sorted_scores = get_sorted_scores(final_scores)
                        
                        # Update the bar graph with the new target_label
                        processed_fig = go.Figure(
                            data=[
                                go.Bar(
                                    x=bar_data["Score"],
                                    y=bar_data["Class"],
                                    orientation="h",
                                    text=None,  # Remove text inside bars
                                    marker=dict(
                                        color=[
                                            "blue" if int(score[0]) == target_label else ("green" if score[1] > 0 else "red")
                                            for score in sorted_scores
                                        ]
                                    ),
                                )
                            ]
                        )     


                        # Retrieve the clicked class for the processed image
                        processed_new_label = int(processed_bar_data["Class"][processed_selected_points[0]["pointIndex"]])

                        # Recompute LRP for the processed image with respect to the new label
                        if st.session_state["dataset_type"] == "original":
                            processed_R, processed_X = compute_LRP(processed_image, processed_new_label,dataset_type="original")

                        elif st.session_state["dataset_type"] == "clean":
                            processed_R, processed_X = compute_LRP(processed_image, processed_new_label,dataset_type="clean")

                with col3:
                    st.markdown(
                        f"<h3 style='text-align: center;'>Label: {true_label} | Pred: {processed_pred_label}</h3>",
                        unsafe_allow_html=True,
                    )

                    # Convert noisy image to PIL format and upscale using LANCZOS
                    processed_image_uint8 = (np.clip(processed_image, 0, 1) * 255).astype("uint8")
                    pil_noisy_image = Image.fromarray(processed_image_uint8)
                    high_res_noisy_image = pil_noisy_image.resize((28 * 4, 28 * 4), Image.LANCZOS)

                    # Plot the resized image
                    processed_fig, ax = plt.subplots(figsize=(1, 1), dpi=150)
                    ax.imshow(high_res_noisy_image, cmap="gray")
                    ax.axis("off")
                    st.pyplot(processed_fig)
                    plt.close(processed_fig)

                # Compute LRP for the processed image
                try:
                    if st.session_state["dataset_type"] == "original":
                        R_processed, X_processed = compute_LRP(processed_image, true_label,dataset_type="original")

                    elif st.session_state["dataset_type"] == "clean":
                        R_processed, X_processed = compute_LRP(processed_image, true_label,dataset_type="clean")


                    if X_processed.ndim == 1:
                        X_processed = X_processed.reshape(28, 28)

                    with col2:
                        st.markdown(f"<h3 style='text-align: center;'>LRP Perturbated Image</h3>", unsafe_allow_html=True)
                        processed_fig, ax = plt.subplots(figsize=(1, 1), dpi=150)
                        LRP_heatmap_with_overlay(R_processed, X_processed, 9, 0.75, interpolation="bicubic", zoom_factor=2, ax=ax)
                        st.pyplot(processed_fig)
                        plt.close(processed_fig)

                except Exception as e:
                    st.error(f"Error generating heatmap for processed image: {e}")


    # Conditionally display the difference heatmap only if user interacted
    if st.session_state.get("user_interacted", False):  # Correct

            with col2:
                try:
                    
                    # Compute LRP for the original image
                    
                    if st.session_state["dataset_type"] == "original":
                        R, X = compute_LRP(selected_image,true_label,dataset_type="original")
                        # Compute LRP for the processed image
                        R_processed, X_processed = compute_LRP(processed_image, true_label,dataset_type="original")

                    elif st.session_state["dataset_type"] == "clean":
                        R, X = compute_LRP(selected_image,true_label,dataset_type="clean")

                    # Compute LRP for the processed image
                    R_processed, X_processed = compute_LRP(processed_image, true_label,dataset_type="clean")

                    

                    # Compute the difference heatmap
                    difference_Metrics = compute_difference_metrics(R, R_processed)

                    # Display the difference heatmap
                    with st.container():
                        

                        # Normalize the difference_heatmap to range [-1, +1]
                        max_abs_value = np.max(np.abs(difference_Metrics["difference_heatmap"]))  # Find the maximum absolute value
                        if max_abs_value > 0:  # Avoid division by zero
                            normalized_heatmap = difference_Metrics["difference_heatmap"] / max_abs_value
                        else:
                            normalized_heatmap = difference_Metrics["difference_heatmap"]  # If all values are zero, leave unchanged

                        # Plot the normalized heatmap
                        processed_fig, ax = plt.subplots(figsize=(5, 5))  # Adjust size as needed
                        st.write("")
                        st.write("")
                        st.markdown(f"<h3 style='text-align: center;'>Difference Heatmap</h3>", unsafe_allow_html=True)
                        plot_difference_heatmap(normalized_heatmap, ax=ax)  # Use the normalized heatmap
                        st.pyplot(processed_fig)
                        plt.close(processed_fig)
                        
                        
                        def styled_metric_display(title, value, title_color="blue", value_color="green", font_size="24px"):
                            title_style = f"""
                            <div style='color: {title_color}; font-size: {font_size}; font-weight: bold;'>
                                {title}
                            </div>
                            """
                            value_style = f"""
                            <div style='color: {value_color}; font-size: {font_size}; font-weight: bold;'>
                                {value}
                            </div>
                            """
                            st.markdown(title_style, unsafe_allow_html=True)
                            st.markdown(value_style, unsafe_allow_html=True)

                        # Display all metrics
                        styled_metric_display("Cosine Similarity:", difference_Metrics["cosine_similarity"])
                        styled_metric_display("SSIM:", difference_Metrics["ssim"])
                        styled_metric_display("MSE:", difference_Metrics["mse"])

                except Exception as e:
                    st.error(f"Error generating difference heatmap: {e}")
            with col1:
                # Define the sentences with HTML styling
                positive_sentence = """
                <p style='font-size: 2em; font-weight: bold; color: blue;'>
                    Positive regions (+) indicate where the model starts paying more attention due to noise or perturbations.
                </p>
                """

                negative_sentence = """
                <p style='font-size: 2em; font-weight: bold; color: red;'>
                    Negative regions (-) show where the model reduces focus, potentially missing important features.
                </p>
                """

                # Display the styled sentences in Streamlit
                st.markdown(positive_sentence, unsafe_allow_html=True)
                st.markdown(negative_sentence, unsafe_allow_html=True)

    
    #Check if the user has drawn something
    if canvas_result.image_data is not None:
        drawn_img = cv2.cvtColor(canvas_result.image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)

        if np.sum(drawn_img) > 0:  # Proceed only if the user has drawn something meaningful
            # Step 1: Resize to 28x28
            resized_img = cv2.resize(drawn_img, (28, 28))

            # Step 2: Center the image
            centered_img = center_image(resized_img)

            # Step 3: Normalize the image
            flattened_img = centered_img.flatten()
            normalized_drawn_img = flattened_img / 255.0

            # Step 4: Predict using the model
            drawn_img_pred = predict_new_image(normalized_drawn_img)

            # Display the "Drawn Image Analysis" expander
            with st.expander("Drawn Image Analysis", expanded=True):
                col1, col2, col3 = st.columns([2, 1, 1])

                # Display the prediction
                with col3:
                    # Center the title and display the drawn image with adjusted size
                    st.markdown(f"<h3 style='text-align: center;'>Predicted Digit: {drawn_img_pred}</h3>", unsafe_allow_html=True)
                    drawn_fig, ax = plt.subplots(figsize=(1, 1), dpi=150)  # Match the size to the LRP heatmap
                    ax.imshow(drawn_img, cmap="gray")
                    ax.axis("off")
                    st.pyplot(drawn_fig)
                    plt.close(drawn_fig)

                # # Display the LRP heatmap for the drawn image
                with col2:
                    try:
                        R_drawn_image, X_drawn_image = compute_LRP(normalized_drawn_img, drawn_img_pred)
                        if X_drawn_image.ndim == 1:
                            X_drawn_image = X_drawn_image.reshape(28, 28)

                        # Center the title and display the LRP heatmap
                        st.markdown(f"<h3 style='text-align: center;'>LRP for Class {drawn_img_pred}</h3>", unsafe_allow_html=True)

                        drawn_fig, ax = plt.subplots(figsize=(1, 1), dpi=150)
                        LRP_heatmap_with_overlay(R_drawn_image, X_drawn_image, 9, 0.75, interpolation="bicubic", zoom_factor=2, ax=ax)
                        st.pyplot(drawn_fig)
                        plt.close(drawn_fig)
                    except Exception as e:
                        st.error(f"Error generating heatmap for drawn image: {e}")

                # Display the class scores for the drawn image
                with col1:
                    st.markdown(f"<h3 style='text-align: center;'>Drawn Image Score WRT Label {drawn_img_pred}</h3>", unsafe_allow_html=True)
                    drawn_scores, drawn_target_label = compute_class_scores(normalized_drawn_img, drawn_img_pred)
                    drawn_sorted_scores = get_sorted_scores(drawn_scores)
                    
                    # Prepare data for the bar chart
                    drawn_bar_data = {
                        "Class": [str(score[0]) for score in drawn_sorted_scores],
                        "Score": [score[1] for score in drawn_sorted_scores],
                    }

                    # Create the bar chart for the drawn image
                    drawn_fig = go.Figure(
                        data=[
                            go.Bar(
                                x=drawn_bar_data["Score"],
                                y=drawn_bar_data["Class"],
                                orientation="h",
                                text=None,
                                marker=dict(
                                    color=[
                                        "blue" if int(score[0]) == drawn_target_label else ("green" if score[1] > 0 else "red")
                                        for score in drawn_sorted_scores
                                    ]
                                ),
                            )
                        ]
                    )
                    drawn_fig.update_traces(hoverinfo="x+y", hoverlabel=dict(font=dict(size=12)), selector=dict(type='bar'))
                    drawn_fig.update_layout(
                    hoverlabel=dict(
                        bgcolor="rgba(0, 0, 0, 1)",  # Black with 80% opacity
                        font=dict(color="white", size=12),  # White text with font size 12
                        namelength=-1,  # Display full class names if needed
                    ),
                    title_font=dict(size=20, color="black"),
                    xaxis=dict(
                        title="Score",
                        title_font=dict(size=16, color="black"),
                        tickfont=dict(size=12, color="black"),
                        zeroline=True,
                        zerolinecolor="black",
                    ),
                    yaxis=dict(
                        title="Class",
                        title_font=dict(size=16, color="black"),
                        tickfont=dict(size=12, color="black"),
                    ),
                    font=dict(color="black"),
                    plot_bgcolor="white",
                    template="simple_white",
                    height=400,
                    margin=dict(l=10, r=10, t=50, b=10),
                )


                    # Display the interactive bar chart
                    drawn_selected_points = plotly_events(drawn_fig , click_event=True)

                    # If the user clicks on a bar, update the graph and LRP
                    if drawn_selected_points:
                        # Retrieve the clicked class (new_label) from the selected bar
                        st.session_state["drawn_bar_clicked"] = True
                        drawn_new_label = int(drawn_bar_data["Class"][drawn_selected_points[0]["pointIndex"]])

                        # Update the title of the bar graph
                        st.markdown(
                            f"<h3 style='text-align: center;'>Drawn Image Score WRT Label {drawn_new_label}</h3>",
                            unsafe_allow_html=True,
                        )

                        # Recalculate the bar graph data
                        drawn_final_scores, drawn_new_label = compute_class_scores(normalized_drawn_img, drawn_new_label)
                        drawn_sorted_scores = get_sorted_scores(drawn_final_scores)
                        new_scores = [score[1] for score in drawn_sorted_scores]
                        new_colors = [
                            "blue" if int(score[0]) == drawn_target_label else ("green" if score[1] > 0 else "red")
                            for score in drawn_sorted_scores
                        ]
                        drawn_fig_new_label = go.Figure(
                            data=[
                                go.Bar(
                                    x=drawn_bar_data["Score"],
                                    y=drawn_bar_data["Class"],
                                    orientation="h",
                                    text=None,  # Remove text inside bars
                                    marker=dict(
                                        color=[
                                            "blue" if int(score[0]) == drawn_target_label else ("green" if score[1] > 0 else "red")
                                            for score in drawn_sorted_scores
                                        ]
                                    ),
                                )
                            ]
                        )

                        drawn_fig_new_label.update_traces(hoverinfo="x+y", hoverlabel=dict(font=dict(size=12)), selector=dict(type='bar'))
                        drawn_fig_new_label.update_layout(
                        hoverlabel=dict(
                            bgcolor="rgba(0, 0, 0, 1)",  # Black with 80% opacity
                            font=dict(color="white", size=12),  # White text with font size 12
                            namelength=-1,  # Display full class names if needed
                        ),
                        title_font=dict(size=20, color="black"),
                        xaxis=dict(
                            title="Score",
                            title_font=dict(size=16, color="black"),
                            tickfont=dict(size=12, color="black"),
                            zeroline=True,
                            zerolinecolor="black",
                        ),
                        yaxis=dict(
                            title="Class",
                            title_font=dict(size=16, color="black"),
                            tickfont=dict(size=12, color="black"),
                        ),
                        font=dict(color="black"),
                        plot_bgcolor="white",
                        template="simple_white",
                        height=400,
                        margin=dict(l=10, r=10, t=50, b=10),
                    )

                        # Update the graph properties
                        drawn_fig_new_label.data[0].x = new_scores
                        drawn_fig_new_label.data[0].marker.color = new_colors

                        # Add cursor styling
                        drawn_fig_new_label.update_traces(hoverinfo="x+y", hoverlabel=dict(font=dict(size=12)), selector=dict(type='bar'))
                        drawn_fig_new_label.update_layout(dragmode=False, hoverlabel=dict(namelength=-1))  # Ensure hand cursor
                        drawn_fig_new_label.update_layout(hoverlabel=dict(font=dict(color="black"))) # make label visible

                        
                        # Display the interactive bar chart
                        drawn_selected_points = plotly_events(drawn_fig_new_label, click_event=True, key="drawn_fig_new_label")

                                        
                        # Recalculate and update the LRP heatmap for the new label
                        R_drawn_image, X_drawn_image = compute_LRP(normalized_drawn_img, drawn_new_label)
                        if X_drawn_image.ndim == 1:
                            X_drawn_image = X_drawn_image.reshape(28, 28)
                        with col2:
                            st.write("")   
                            st.write("")   
                            st.write("")   
                            st.write("")   
                            
                            
                            
                        with col2:
                            st.markdown(f"<h3 style='text-align: center;'>LRP for Class {drawn_new_label}</h3>", unsafe_allow_html=True)
                            drawn_fig, ax = plt.subplots(figsize=(1, 1), dpi=150)
                            LRP_heatmap_with_overlay(R_drawn_image, X_drawn_image, 9, 0.75, interpolation="bicubic", zoom_factor=2, ax=ax)
                            st.pyplot(drawn_fig)
                            plt.close(drawn_fig)

import streamlit as st

with tab2:
    st.subheader("Confidence Value Occurrences for All Examples (Train and Test for both Original & Clean dataset)")

    with st.expander("Confidence Value Occurrences for All Examples(Train & Test for both Original and Clean dataset)", expanded=True):
        # Create two equal columns
        col1, col2 = st.columns([1, 1])

        with col1:
            # Display the train dataset confidence distribution image
            st.image(
                "Confidence_Misclassified/overall/train_misclassified_confidence_distribution_4_percent_bins.png", 
                caption="Train Dataset Confidence Distribution"
            )

        with col2:
            # Display the test dataset confidence distribution image
            st.image(
                "Confidence_Misclassified/overall/test_misclassified_confidence_distribution_4_percent_bins.png", 
                caption="Test Dataset Confidence Distribution"
            )
    # Second Expander
    st.header("Confidence value distributions for misclassified examples WRT gradually PGD Attacks")
    with st.expander("Confidence value distributions for misclassified examples WRT gradually PGD Attacks", expanded=True):
         st.image(
                "confidence_WRT_attacks/confidence_drop_plot.png",
                caption="Confidence Drop Plot"
            )
           