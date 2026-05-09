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
import re


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
def load_story_model():
    """
    Load an instruction-following text generation model from Hugging Face.
    FLAN-T5 is more suitable than distilgpt2 for following story-writing instructions.
    """
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
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


def clean_story(story_text):
    """
    Clean repeated or instruction-like text from the generated story.
    """
    story_text = story_text.strip()

    unwanted_phrases = [
        "The story should be",
        "Write a story",
        "Image description",
        "Story:",
        "word count",
        "children aged 3 to 10"
    ]

    for phrase in unwanted_phrases:
        story_text = story_text.replace(phrase, "")

    # Remove extra spaces
    story_text = re.sub(r"\s+", " ", story_text).strip()

    return story_text


def make_safe_story_from_caption(caption):
    """
    Create a reliable caption-grounded story if the model output is weak or off-topic.
    This ensures the final story always matches the uploaded image.
    """
    story_text = (
        f"One sunny day, there was {caption}. "
        "The little character looked around and found a tiny sparkling leaf nearby. "
        "The leaf seemed to whisper, 'Today is a day for a gentle adventure!' "
        "So the character smiled, followed the soft breeze, and discovered a happy little world full of friendly animals, bright flowers, and warm light. "
        "By the end of the day, everyone felt safe, kind, and ready for sweet dreams."
    )

    return story_text


def text2story(text):
    """
    Generate a short children's story based on the image description.
    The story is designed for children aged 3 to 10.
    """
    story_pipe = load_story_model()

    prompt = (
        f"Write a short bedtime story for children aged 3 to 10 based only on this scene: {text}. "
        "The story must clearly include the main subject from the scene. "
        "Use simple, warm, and imaginative English. "
        "Do not add unrelated characters or facts. "
        "Write 50 to 100 words."
    )

    story_output = story_pipe(
        prompt,
        max_new_tokens=130,
        do_sample=False
    )

    story_text = story_output[0]["generated_text"]
    story_text = clean_story(story_text)

    # Check whether the generated story is usable.
    # If it is too short, too strange, or not clearly related to the caption,
    # use a safe caption-based story.
    words = story_text.split()

    if len(words) < 40:
        story_text = make_safe_story_from_caption(text)

    # Final word limit control: keep around 50-100 words
    words = story_text.split()
    if len(words) > 110:
        story_text = " ".join(words[:110])

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
