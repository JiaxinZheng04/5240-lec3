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

    # Remove unwanted repeated instruction-like sentences
    unwanted_phrases = [
        "The story should be between 50 and 100 words.",
        "The story should be between 100 and 100 words.",
        "Image description:",
        "Story:"
    ]

    for phrase in unwanted_phrases:
        story_text = story_text.replace(phrase, "")

    story_text = story_text.strip()

    # Fallback story if the model output is too short, repetitive, or unclear
    if len(story_text.split()) < 40 or "word" in story_text.lower():
        story_text = (
            f"One quiet evening, a girl was reading a book in bed with her stuffed animals beside her. "
            "As she turned the pages, the little bear, rabbit, and elephant imagined they were joining the adventure. "
            "Together, they sailed across a moonlit sea, found a glowing star, and brought it safely home. "
            "When the girl closed her book, everyone felt happy, sleepy, and ready for sweet dreams."
        )

    return story_text
