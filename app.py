import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. Load your trained model and define labels ---
model = tf.keras.models.load_model("kidney_model_unbiased_v1.h5")
labels = ['Cyst', 'Normal', 'Stone', 'Tumor']

# --- 2. Define the prediction function ---
def predict(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction_array = model.predict(image)[0]
    confidences = {labels[i]: float(prediction_array[i]) for i in range(len(labels))}
    return confidences

# --- 3. Custom CSS for Glassmorphism and 3D Text Effect ---
css = """
body {
    background-image: url('https://images.unsplash.com/photo-1531685250784-7569952593d2?q=80&w=2874&auto=format&fit=crop');
    background-size: cover;
}
.gradio-container { background: none; }
.gr-panel {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}
#title {
    color: white;
    font-size: 40px;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.6);
}
#description {
    color: #E0E0E0;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.5);
}
"""

# --- 4. Build the UI with gr.Blocks ---
with gr.Blocks(css=css) as demo: # <-- THEME HAS BEEN REMOVED HERE
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("# Kidney Disease Detector", elem_id="title")
            gr.Markdown("Upload a kidney CT scan to detect the disease type (Cyst, Normal, Stone, or Tumor).", elem_id="description")
        with gr.Column(scale=1):
            gr.Markdown("""
            <iframe title="Kidney Anatomy 3D" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share src="https://sketchfab.com/models/c391fd03fa9240448a600d40f5c1d799/embed" style="width: 100%; height: 250px;"></iframe>
            """)

    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(type="numpy", label="Upload Kidney CT Scan")
            submit_btn = gr.Button("Submit")
        with gr.Column(scale=1):
            output_label = gr.Label(num_top_classes=4, label="Prediction")
            
    gr.Examples(
        examples=[
        ["cyst_example_1.jpg"],
        ["cyst_example_2.jpg"],
        ["normal_example_1.jpg"],
        ["normal_example_2.jpg"],
        ["stone_example_1.jpg"],
        ["stone_example_2.jpg"],
        ["tumor_example_1.jpg"],
        ["tumor_example_2.jpg"]
    ],
        inputs=input_image
    )
    
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=output_label
    )

# --- 5. Launch the app ---
demo.launch()

