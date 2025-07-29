import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model dan tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("model_sarkas")
    model = AutoModelForSequenceClassification.from_pretrained("model_sarkas")
    return tokenizer, model

tokenizer, model = load_model()

st.title("Deteksi Sarkasme Komentar TikTok 🇮🇩")

text = st.text_area("Masukkan komentar TikTok:", "")

if st.button("Deteksi Sarkasme"):
    if text.strip() == "":
        st.warning("Masukkan komentar terlebih dahulu.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()

        label = "Sarkastik 😏" if pred == 1 else "Tidak Sarkastik 🙂"
        st.success(f"Hasil Deteksi: **{label}**")
