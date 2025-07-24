import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. Load your trained model and define labels ---
model = tf.keras.models.load_model("kidney_tumor_model.h5")
labels = ['Cyst', 'Normal', 'Stone', 'Tumor']

# --- 2. Define the prediction function ---
def predict(image):
    # Process the image for the model
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    prediction_array = model.predict(image)[0]
    
    # Format the output for the Label component
    confidences = {labels[i]: float(prediction_array[i]) for i in range(len(labels))}
    return confidences

# --- 3. Create a Custom Theme ---
# We'll stick with the Soft theme, as it supports both light and dark modes.
theme = gr.themes.Soft(
    primary_hue="red",
    secondary_hue="pink",
)

# --- 4. Custom CSS for a Light Theme Appearance ---
css = """
/* Using a direct URL is more reliable for background images */
.gradio-container {
    background-image: url('http://googleusercontent.com/image_generation_content/4');
    background-size: cover;
    background-position: center;
}

/* Glassmorphism panel style adjusted for a light background */
.gr-panel {
    background: rgba(255, 255, 255, 0.4); /* Made it more opaque white for readability */
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.25);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
}

/* Text colors changed to be dark, so they are visible on a light background */
#title {
    color: #4a4a4a; /* Dark gray for the title */
    font-size: 40px;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.7);
}
#description {
    color: #575757; /* Slightly lighter gray for description */
    text-shadow: 1px 1px 2px rgba(255,255,255,0.5);
}
"""

# --- 5. Build the UI with the Theme and CSS ---
with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown("# Kidney Disease Detector", elem_id="title")
    gr.Markdown("Upload a kidney CT scan to detect the disease type (Cyst, Normal, Stone, or Tumor).", elem_id="description")

    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(type="numpy", label="Upload Kidney CT Scan")
            submit_btn = gr.Button("Submit")
        with gr.Column(scale=1):
            output_label = gr.Label(num_top_classes=4, label="Prediction")
            
    gr.Examples(
        examples=[
            "cyst_example_1.jpg", "cyst_example_2.jpg",
            "normal_example_1.jpg", "normal_example_2.jpg",
            "stone_example_1.jpg", "stone_example_2.jpg",
            "tumor_example_1.jpg", "tumor_example_2.jpg"
        ],
        inputs=input_image,
        label="Examples" # Added a label for clarity
    )
    
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=output_label
    )

# --- 6. Launch the app ---
demo.launch()