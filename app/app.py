import streamlit as st
import joblib
import os
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1) Repo HF của bạn
repo_id = "Salaghati/moodsense-distilbert"

# 2) Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model     = AutoModelForSequenceClassification.from_pretrained(
    repo_id,
    problem_type="multi_label_classification",
)
model.eval()

# 3) Đưa lên device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

st.set_page_config(page_title="MoodSense by Tu", page_icon="🧠", layout="centered")

st.image("https://i.imgur.com/QzloMmO.png", width=120)
st.markdown("## 👋 Hi! How are you feeling today?")


model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
if os.path.exists(model_path):
    model = joblib.load(model_path)
    
    st.markdown("### ✍️ Enter your journal entry below:")
    journal_text = st.text_area("", height=200, placeholder="Today I feel...")

    if st.button("🔮 Predict Mood"):
        if journal_text.strip():
            prediction = model.predict([journal_text])[0]
            st.success(f"📝 Predicted Mood: **{prediction}**")

            if prediction == "Happy":
                st.balloons()
                st.info("💛 Keep happy nha Tatbolz")
            elif prediction == "Sad":
                st.warning("💙 Take care of yourself. You're not alone nhe Tatbolz")
            elif prediction == "Anxious":
                st.info("🧘‍♂️ Try some breathing exercises. You're doing great!")
            else:
                st.info("🌤️ Steady and calm is good too.")
        else:
            st.warning("Please enter something to analyze.")
else:
    st.error("❌ Model file not found. Please train the model and upload model.pkl.")

with st.sidebar:
    st.image("app/avatar.jpg", width=100)
    st.markdown("### 🤖 MoodSense")
    st.markdown("Built by Tu – I love AI")
    st.markdown("[GitHub Repo](https://github.com/salaghat1/MoodSense)")
    st.markdown("— Powered by Streamlit & scikit-learn —")
