import tensorflow as tf
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np

# Streamlit page configuration
st.set_page_config(
    page_title="Plant Disease Classification",
    layout="wide",
    page_icon="🌱"
)

# Function to load the model
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

# Load Potato Model
model_path = "../Potato Disease Classification/Saved_models/model_v1_potato.h5"
model = load_model(model_path)
model_path2 ="../Potato Disease Classification/Saved_models/Saved_models1_tomato.h5"
model2= load_model(model_path2)
# Class names for the potato model
POTATO_CLASSES = ["Potato__Early_blight", "Potato__Late_blight", "Potato__Healthy"]
TOMATO_CLASSES = [
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]
# Sidebar menu for plant selection
with st.sidebar:
    selected = option_menu(
        "Plant Disease Outbreak",
        ["Potato", "Tomato"],  # Add "Tomato" as a placeholder
        menu_icon=" ",
        icons=['leaf', 'apple-alt'],
        default_index=0
    )

if selected == "Potato":
    st.title("Predicting Potato Disease")
    upload_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if upload_file is not None:
        try:
            # Display uploaded image
            image = Image.open(upload_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            # Preprocess the image
            image = image.convert("RGB")
            image = image.resize((256, 256))
            image = np.array(image) / 255.0
            img_array = tf.expand_dims(image, 0)

            # Model prediction
            predictions = model.predict(img_array)
            predicted_class = POTATO_CLASSES[np.argmax(predictions[0])]
            confidence = round(100 * (np.max(predictions[0])), 2)

            # Display results
            st.write("")
            st.write("Classifying...")
            st.write(f"**Predicted Class:** {predicted_class}")
            st.write(f"**Confidence:** {confidence}%")
            st.success("Prediction completed successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif selected == "Tomato":
    st.title("Tomato Disease Classification ")
    upload_file=st.file_uploader("Choose an image...",type=['jpg','png','jpeg'])
    if upload_file is not None:
        try:
            image =Image.open(upload_file)
            st.image(image,caption="Uploaded Image.",use_column_width=True)
            
            image=image.convert("RGB")
            image=image.resize((256,256))
            image = np.array(image)/255.0
            img_array=tf.expand_dims(image,0)
            
            predictions = model.predict(img_array)
            predicted_class= TOMATO_CLASSES[np.argmax(predictions[0])]
            confidence= round(100* (np.max(predictions[0])),2)
            
            st.write("Classifing..")
            st.write(f"Predicted Class: {predicted_class}")
            st.write(f" Confidence: {confidence}%")
            st.success("Prediction Completed successfully")
        except Exception as e:
            st.error(f"An error Occurred: {e}")
            