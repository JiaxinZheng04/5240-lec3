# Program title: Storytelling App
# ISOM5240 Individual Assignment
# This app turns an uploaded image into a short children's story and audio.

# -----------------------------
# Import part
# -----------------------------
import streamlit as st
from transformers import pipeline
from PIL import Image
import tempfile
import os


# -----------------------------
# Model loading part
# -----------------------------

@st.cache_resource
def load_img2text_model():
    """
    Load the image-to-text model from Hugging Face.
    This model generates a caption based on the uploaded image.
    """
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )


@st.cache_resource
def load_audio_model():
    """
    Load the text-to-audio model from Hugging Face.
    This model converts the generated story into speech.
    """
    return pipeline(
        "text-to-audio",
        model="Matthijs/mms-tts-eng"
    )


# -----------------------------
# Function part
# -----------------------------

def img2text(image_path):
    """
    Convert an uploaded image into a text description.
    """
    image_to_text_model = load_img2text_model()
    text = image_to_text_model(image_path)[0]["generated_text"]
    return text


def text2story(text):
    """
    Generate a short, child-friendly story based directly on the image caption.
    This version avoids repeated or unrelated content.
    """

    # Clean the image caption
    scene = text.strip().lower()

    story_text = (
        f"One peaceful morning, {scene}. "
        "The little character looked around and noticed the world was full of tiny wonders. "
        "A soft breeze moved through the leaves, and a friendly bird came to say hello. "
        "Together, they imagined a gentle adventure above the trees, where every cloud looked like a dream. "
        "When the sun began to set, the character smiled, feeling brave, happy, and ready to share the story with friends."
    )

    return story_text


def text2audio(story_text):
    """
    Convert the generated story into audio data.
    """
    audio_pipe = load_audio_model()
    audio_data = audio_pipe(story_text)
    return audio_data


# -----------------------------
# Main part
# -----------------------------

st.set_page_config(
    page_title="Your Image to Audio Story",
    page_icon="📖",
    layout="centered"
)

st.title("📖 Turn Your Image to Audio Story")

st.write(
    "Upload an image, and this app will generate a short children's story and read it aloud."
)

uploaded_file = st.file_uploader(
    "Select an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Save uploaded image locally as a temporary file
    bytes_data = uploaded_file.getvalue()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(bytes_data)
        image_path = temp_file.name

    # Display uploaded image
    image = Image.open(image_path)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Story"):

        # Stage 1: Image to Text
        st.text("Processing image to text...")
        scenario = img2text(image_path)
        st.write(f"**Description:** {scenario}")

        # Stage 2: Text to Story
        st.text("Generating a story...")
        story = text2story(scenario)
        st.write(f"**Story:** {story}")

        # Stage 3: Story to Audio
        st.text("Generating audio data...")
        audio_data = text2audio(story)

        # Play audio directly using Streamlit
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)

    # Remove temporary image file
    if os.path.exists(image_path):
        os.remove(image_path)
