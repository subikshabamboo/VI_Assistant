import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pyttsx3


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css") 


st.markdown("<h1 style='text-align: center;'>Image Assistant for Visually Impaired People</h1>", unsafe_allow_html=True)
st.markdown("<div class='stText'>Upload an image and get to know what's in it!</div>",unsafe_allow_html=True)

uploaded_file=st.file_uploader("Choose an Image",type=["jpg","png","jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Generating Caption...")

    # Load model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Generate caption
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    # Display caption
    st.success(f"Caption: {caption}")

    # Speak caption
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Use female voice
    engine.say(caption)
    engine.runAndWait()

st.markdown("<div class='footer'>© 2025 Subiksha MV. All rights reserved.</div>", unsafe_allow_html=True)

