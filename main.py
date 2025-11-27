import streamlit as st 
import tensorflow as tf
import numpy as np
import webbrowser

# TensorFlow model prediction
def model_prediction(test_image):
    cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    input_arr = input_arr / 255.0      # Normalize if model was trained this way
    predictions = cnn.predict(input_arr)
    result_index = np.argmax(predictions)
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home page
if app_mode == "Home":
    st.header("SMART PLANT DISEASE DETECTION AND FORECASTING SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the SMART PLANT DISEASE DETECTION AND
FORECASTING SYSTEM! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases.

    ### How It Works
    1. **Upload Image** on the **Disease Recognition** page.
    2. **Analysis** using our deep learning model.
    3. **Results** with disease name and recommendations.

    ### Why Choose Us?
    - **Accurate:** State-of-the-art ML model.
    - **Easy to Use:** Friendly UI.
    - **Fast:** Results in seconds.

    ### Get Started
    Go to **Disease Recognition** in the sidebar to begin!

    ### About Us
    Learn more on the **About** page.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation. The original dataset is on GitHub.

    It contains ~87,000 RGB images of healthy and diseased leaves, categorized into 38 classes.

    #### Contents:
    - `train/` — 70,295 images
    - `validation/` — 17,572 images
    - `test/` — 33 images (for evaluation)
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")

    st.markdown("""Want to browse extra images? [Click here](https://google.com)""", unsafe_allow_html=True)

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_container_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction:")

            try:
                result_index = model_prediction(test_image)

                # Disease class labels
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]

                st.success(f"Model is predicting it's a **{class_name[result_index]}**.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


