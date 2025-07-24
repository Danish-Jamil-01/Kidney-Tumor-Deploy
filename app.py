import gradio as gr
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np

# --- 1. Recreate the Model Architecture ---
# This function defines a standard CNN structure.
def create_model():
    model = models.Sequential([
        # The input shape matches the size your images are resized to (64x64 with 3 color channels)
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        # The output layer has 4 units for your 4 classes
        layers.Dense(4, activation='softmax')
    ])
    return model

# --- 2. Load the Model Weights ---
# Create the model structure
model = create_model()
# Load ONLY the saved weights into this structure, ignoring the old optimizer/config
model.load_weights('kidney_tumor_model.h5')
# Compile the model with a modern optimizer so it can be used for predictions
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the class labels
labels = ['Cyst', 'Normal', 'Stone', 'Tumor']

# --- 3. Define the Prediction Function ---
def predict(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction_array = model.predict(image)[0]
    confidences = {labels[i]: float(prediction_array[i]) for i in range(len(labels))}
    return confidences

# --- 4. Create a Custom Theme and CSS ---
theme = gr.themes.Soft(primary_hue="red", secondary_hue="pink")
css = """
.gradio-container {
    background-image: url('/file=background.png');
    background-size: cover;
    background-position: center;
}
.gr-panel {
    background: rgba(255, 255, 255, 0.1); 
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px); 
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}
#title { color: white; font-size: 40px; text-shadow: 2px 2px 8px rgba(0,0,0,0.7); }
#description { color: #E0E0E0; text-shadow: 1px 1px 4px rgba(0,0,0,0.6); }
"""

# --- 5. Build the UI ---
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
        label="Examples"
    )

    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=output_label
    )

# --- 6. Launch the App ---
demo.launch()