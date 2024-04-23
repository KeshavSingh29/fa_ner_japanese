import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)
from app.services.utils import inference, token_aggregator


# Load best Model
st.info("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
inference_model = AutoModelForTokenClassification.from_pretrained("app/ner_model")
st.success("Model loaded successfully!")

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
    annotated_html = annotate_text(*result)

    # Display results
    st.write("Named Entities Found:")
    st.markdown(annotated_html, unsafe_allow_html=True)
