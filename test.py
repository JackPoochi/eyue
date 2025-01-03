import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from PIL import Image
import matplotlib.pyplot as plt

# Function to load the model with error handling
def load_model_with_fallback(model_url):
    try:
        # Download the model file to a local directory
        model_path_local = get_file("model1.h5", model_url)
        model = load_model(model_path_local)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.write(
            "Ensure that the TensorFlow/Keras version matches the version used to save the model. "
            "Consider re-saving the model or contacting the model provider."
        )
        return None

# Preprocessing functions
def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size)

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def adjust_white_balance(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(result)
    l = cv2.equalizeHist(l)
    result = cv2.merge((l, a, b))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def preprocess_image(image):
    resized = resize_image(image)
    enhanced = enhance_contrast(resized)
    balanced = adjust_white_balance(enhanced)
    return balanced

# Google Drive model file URL (direct download link)
MODEL_URL = "https://drive.google.com/uc?id=1MjgxZyWDK64ptQp5jI_ZDpDpKBDwI9-E"

# Load the model
model = load_model_with_fallback(MODEL_URL)

# Streamlit app
st.title("Dry Eye Severity Grading and Image Classification")

# Upload image
uploaded_file = st.file_uploader("Upload an image of the eye for classification:", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    processed_image = preprocess_image(image)

    # Display original and processed images
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(processed_image)
    ax[1].set_title("Processed Image")
    ax[1].axis("off")

    st.pyplot(fig)

# Questionnaire
questions = [
    "Is your eyes sensitive to light?",
    "Does your eyes feel gritty?",
    "Does your eyes feel painful or sore?",
    "Do you have blurred vision?",
    "Do you have poor vision?",
    "Do you read books daily?",
    "Do you drive car or bike at night?",
    "Do you work with computer?",
    "Daily screen timing?",
    "Had any issues with eyes in windy condition?",
    "Does your eyes feel very dry in places with low humidity?",
    "Do you ever feel uncomfortable or have problems when you're in places with air conditioning?"
]

options = ["None of the time", "Some of the time", "Half of the time", "Most of the time", "All the time"]
scores = {"None of the time": 0, "Some of the time": 1, "Half of the time": 2, "Most of the time": 3, "All the time": 4}

responses = []
for question in questions:
    response = st.radio(question, options, key=question)
    responses.append(scores[response])

# Recommendations with medicines and Ayurvedic remedies
recommendations = {
    "Eyelid Labels": {
        "Recommendation": "Consult an ophthalmologist for eyelid hygiene.",
        "Medicine": "Erythromycin Eye Ointment or Tetracycline Eye Drops",
        "Ayurvedic Remedy": "Apply Triphala Eye Wash or Aloe Vera Gel.",
        "Survival Period": "Improvement expected within 2-3 weeks with proper care."
    },
    "Meibomian Gland Labels": {
        "Recommendation": "Consider warm compress therapy.",
        "Medicine": "Azithromycin Eye Drops or Doxycycline Tablets",
        "Ayurvedic Remedy": "Use warm castor oil packs or massage with Brahmi oil.",
        "Survival Period": "Symptom relief usually seen within 3-4 weeks."
    },
    "Original Images": {
        "Recommendation": "Maintain regular eye check-ups.",
        "Medicine": "Lubricating Eye Drops (Artificial Tears)",
        "Ayurvedic Remedy": "Consume Triphala Churna with honey for eye health.",
        "Survival Period": "Continuous care recommended for long-term benefits."
    }
}

# Predict button
if st.button("Predict"):
    if uploaded_file is not None and model is not None:
        # Calculate questionnaire score
        total_score = sum(responses)
        st.write(f"Total Questionnaire Score: {total_score}")

        # Severity grading
        if total_score >= 30:
            severity = "Extreme Dry Eyes"
        elif 25 <= total_score < 30:
            severity = "Moderate Dry Eyes"
        elif 15 <= total_score < 25:
            severity = "Mild Dry Eyes"
        else:
            severity = "No Dry Eyes"

        st.write(f"Severity Grade: {severity}")

        # Image classification
        processed_image_resized = resize_image(processed_image)
        input_image = np.expand_dims(processed_image_resized, axis=0) / 255.0
        prediction = model.predict(input_image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        class_labels = ["Eyelid Labels", "Meibomian Gland Labels", "Original Images"]
        selected_recommendation = recommendations[class_labels[predicted_class]]

        # Display classification and recommendations
        st.write(f"Image Classification: {class_labels[predicted_class]}")
        st.write(f"Recommendation: {selected_recommendation['Recommendation']}")
        st.write(f"Suggested Medicine: {selected_recommendation['Medicine']}")
        st.write(f"Ayurvedic Remedy: {selected_recommendation['Ayurvedic Remedy']}")
        st.write(f"Estimated Survival Period: {selected_recommendation['Survival Period']}")
    else:
        st.warning("Please upload an image and ensure the model is loaded.")

# Footer
st.write("---")
st.write("Developed by PEMCHIP")
