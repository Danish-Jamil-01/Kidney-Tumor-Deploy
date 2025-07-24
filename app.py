import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model("kidney_tumor_model.h5")

# Define the labels for your 4 classes
# Make sure the order matches how you trained your model: 0=Cyst, 1=Normal, 2=Stone, 3=Tumor
labels = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Create the prediction function
def predict(image):
    # Resize the image to 64x64, as required by your model
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize((64, 64))
    image = np.array(image) / 255.0  # Normalize the image

    # The model expects a batch of images, so add a batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Make a prediction
    prediction_array = model.predict(image)[0]
    
    # Create a dictionary of labels and their probabilities
    confidences = {labels[i]: float(prediction_array[i]) for i in range(len(labels))}
    
    return confidences

# Create the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Kidney CT Scan"),
    outputs=gr.Label(num_top_classes=4, label="Prediction"),
    title="Kidney Disease Detector",
    description="Upload a kidney CT scan to detect the disease type. This demo uses a Convolutional Neural Network trained on a public dataset.",
    examples=[
        ["cyst_example_1.jpg"],
        ["cyst_example_2.jpg"],
        ["normal_example_1.jpg"],
        ["normal_example_2.jpg"],
        ["stone_example_1.jpg"],
        ["stone_example_2.jpg"],
        ["tumor_example_1.jpg"],
        ["tumor_example_2.jpg"]
    ]
)

# Launch the app
iface.launch()