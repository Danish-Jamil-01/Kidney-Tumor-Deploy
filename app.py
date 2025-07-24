import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# --- Your model loading and predict function stay the same ---
model = tf.keras.models.load_model("kidney_tumor_model.h5")
labels = ['Cyst', 'Normal', 'Stone', 'Tumor']

def predict(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction_array = model.predict(image)[0]
    confidences = {labels[i]: float(prediction_array[i]) for i in range(len(labels))}
    return confidences
# -----------------------------------------------------------


# --- Define the new UI using gr.Blocks ---
with gr.Blocks(theme='soft') as demo:
    gr.Markdown("# Kidney Disease Detector") # Main title
    gr.Markdown("Upload a kidney CT scan to detect the disease type (Cyst, Normal, Stone, or Tumor).") # Description

    with gr.Row():
        with gr.Column():
            # Input components
            input_image = gr.Image(label="Upload Kidney CT Scan")
            submit_btn = gr.Button("Submit")

        with gr.Column():
            # Output components
            output_label = gr.Label(num_top_classes=4, label="Prediction")

    # Examples
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
        inputs=input_image # Tell the examples which component to populate
    )
    
    # Define what happens when the button is clicked
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=output_label
    )

# Launch the new demo
demo.launch()