import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. Load your trained model and define labels ---
model = tf.keras.models.load_model("kidney_tumor_model.h5")
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

# --- 3. Create a Custom Theme for Colors ---
theme = gr.themes.Soft(
    primary_hue="red",
    secondary_hue="pink",
)

# --- 4. Build the UI with the Theme ---
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# Kidney Disease Detector")
    gr.Markdown("Upload a kidney CT scan to detect the disease type (Cyst, Normal, Stone, or Tumor).")

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

# --- 5. Launch the app with sharing enabled ---
demo.launch(share=True)