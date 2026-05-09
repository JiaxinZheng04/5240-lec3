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
def load_story_model():
    """
    Load the text-generation model from Hugging Face.
    This model expands the image caption into a short children's story.
    """
    return pipeline(
        "text-generation",
        model="distilgpt2"
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
    Generate a short children's story based on the image description.
    The story is designed for children aged 3 to 10.
    """
    story_pipe = load_story_model()

    prompt = (
        "You are a children's storyteller.\n"
        "Write one short, warm, and imaginative bedtime story for children aged 3 to 10.\n"
        "Use simple English. Do not repeat the instruction. Do not mention word count.\n"
        f"Image description: {text}\n"
        "Story:\n"
    )

    story_output = story_pipe(
        prompt,
        max_new_tokens=110,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3
    )

    generated_text = story_output[0]["generated_text"]

    # Keep only the part after "Story:"
    story_text = generated_text.split("Story:")[-1].strip()

    # Remove unwanted instruction-like phrases
    unwanted_phrases = [
        "The story should be between 50 and 100 words.",
        "The story should be between 100 and 100 words.",
        "Image description:",
        "Story:",
        "Do not repeat the instruction.",
        "Do not mention word count."
    ]

    for phrase in unwanted_phrases:
        story_text = story_text.replace(phrase, "")

    story_text = story_text.strip()

    # Fallback story if the model output is too short, repetitive, or unclear
    if len(story_text.split()) < 40 or "word" in story_text.lower():
        story_text = (
            f"One quiet evening, there was a lovely scene: {text}. "
            "A curious child looked at it and imagined a magical adventure. "
            "The toys nearby came to life and joined the journey. "
            "Together, they explored a bright moonlit world, helped a lost star find its way home, "
            "and learned that kindness can make every adventure special. "
            "At the end of the night, everyone smiled and fell asleep with happy dreams."
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
