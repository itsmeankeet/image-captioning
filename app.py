import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

# Import custom classes
from utils.model_architecture import (
    ImageCaptioningModel, LRSchedule,
    TransformerEncoderBlock, TransformerDecoderBlock, PositionalEmbedding
)
from tensorflow.keras.layers import TextVectorization

# ---- Custom standardization function (EDIT to match your notebook) ----
# Define the custom standardization function
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    strip_chars = "!\"#$%&'()*+,-./:;=?@[\]^_`{|}~1234567890"
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

# Load the vectorizer from saved config and vocab
def load_vectorizer(config_path, vocab_path):
    # Load config and vocabulary
    with open(config_path, 'r') as f:
        config = json.load(f)
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    # Remove fields not accepted by from_config
    config['standardize'] = None
    config.pop('batch_input_shape', None)

    # Reconstruct vectorizer
    vectorizer = TextVectorization.from_config(config)
    vectorizer.standardize = custom_standardization
    vectorizer.set_vocabulary(vocab)

    return vectorizer

# ---- Load model and vocab ----
@st.cache_resource(show_spinner=False)
def load_model_and_vocab():
    
    custom_objects = {
        "ImageCaptioningModel": ImageCaptioningModel,
        "LRSchedule": LRSchedule,
        "TransformerEncoderBlock": TransformerEncoderBlock,
        "TransformerDecoderBlock": TransformerDecoderBlock,
        "PositionalEmbedding": PositionalEmbedding,
    }
    model = tf.keras.models.load_model('./models/best_model.keras', custom_objects=custom_objects, compile=False)
    vectorizer = load_vectorizer('./models/vectorization_config.json', './models/vocabulary.json')
    vocab = vectorizer.get_vocabulary()
    index_to_word = {idx: word for idx, word in enumerate(vocab)}
    return model, index_to_word, vectorizer

# ---- Preprocess image ----
def decode_and_resize(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

# ---- Greedy search ----
def greedy(image_path, model, vectorizer, index_to_word, max_len=24):
    # Preprocess
    image = decode_and_resize(image_path)
    image = tf.expand_dims(image, 0)
    # EfficientNet encoder, adapt if needed!
    image_features = model.cnn_model(image) if hasattr(model, 'cnn_model') else image
    encoded_img = model.encoder(image_features, training=False)
    decoded_caption = "<start>"
    for i in range(max_len):
        tokenized_caption = vectorizer([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        preds = model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)
        sampled_token_index = np.argmax(preds[0, i, :])
        sampled_token = index_to_word[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token
    decoded_caption = decoded_caption.replace("<start>", "").replace("<end>", "").strip()
    return decoded_caption

# ---- Streamlit UI ----
st.title("🖼️ Image Captioning Demo (Flickr30k, Transformers)")

model, index_to_word, vectorizer = load_model_and_vocab()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    # Save the file temporarily
    tmp_path = "temp_uploaded_image.jpg"
    img.save(tmp_path)
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            caption = greedy(tmp_path, model, vectorizer, index_to_word)
            st.success(f"**Generated Caption:** {caption}")
    # Cleanup
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
