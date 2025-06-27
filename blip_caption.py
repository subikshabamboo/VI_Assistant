from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import pyttsx3

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image (you can replace this with your own file path)
image_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')

# Process image and generate caption
inputs = processor(images=image, return_tensors="pt")
output = model.generate(**inputs)
caption = processor.decode(output[0], skip_special_tokens=True)


print("Generated Caption:", caption)


engine=pyttsx3.init()
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)
engine.say(caption)
engine.runAndWait()
