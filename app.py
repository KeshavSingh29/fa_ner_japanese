import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)
from app.services.utils import inference, token_aggregator


# Load best Model
st.info("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
try:
    inference_model = AutoModelForTokenClassification.from_pretrained("ner_model")
    st.success("Model loaded successfully!")
except Exception as e:
    st.warning("Model not found, please download the weights or train the model first")

entity_colors = {
    '政治的組織名': "red",
    '施設名': "blue",
    '法人名': "orange",
    '地名': "green",
    '製品名': "purple",
    'その他の組織名': "brown",
    '人名': "pink",
    'イベント名': "cyan"
}


def annotate_text(*annotations):
    """Helper function to display text with annotations"""
    annotated_text = ""
    for annotation in annotations:
        tag = annotation['tag']
        text = annotation['text']
        
        if tag != 'O':  # Highlight tagged text
            color = entity_colors.get(tag, "black")  # Get color based on entity type
            annotated_text += f'<mark style="background-color: {color}; color: white;">{text} <sup>({tag})</sup></mark> '
        else:
            annotated_text += text
    return annotated_text

st.title("Named Entity Recognition (Japanese Text)")

# User input
sentence = st.text_input("Enter a sentence:")

# Button to perform NER
if st.button("Extract NER Tags"):
    # Perform NER on the input sentence
    raw_labels = inference(data_piece=sentence, inference_model=inference_model, tokenizer=tokenizer)
    raw_tokens = tokenizer.tokenize(sentence)
    result = token_aggregator(tokens=raw_tokens, labels=raw_labels)
    only_entities = [x for x in result if x['tag'] != "O"]
    annotated_html = annotate_text(*result)

    # Display results
    st.write(f"Named Entities Found: {len(only_entities)}")
    st.markdown(annotated_html, unsafe_allow_html=True)
