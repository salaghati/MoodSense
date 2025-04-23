# app/app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ─── 0) Page setup ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoodSense by Tu",
    page_icon="🧠",
    layout="centered"
)
st.image("https://i.imgur.com/QzloMmO.png", width=120)
st.markdown("## 👋 Hi! How are you feeling today?")

# ─── 1) Load model & tokenizer from HF Hub ────────────────────────────────────
repo_id  = "Salaghati/moodsense-distilbert"
hf_token = st.secrets["HF_TOKEN"]

tokenizer = AutoTokenizer.from_pretrained(
    repo_id,
    use_auth_token=hf_token,
)
model = AutoModelForSequenceClassification.from_pretrained(
    repo_id,
    problem_type="multi_label_classification",
    use_auth_token=hf_token,
)
model.eval()

# ─── 2) Device ────────────────────────────────────────────────────────────────
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# ─── 3) Text input ────────────────────────────────────────────────────────────
journal_text = st.text_area(
    "✍️ Enter your journal entry below:",
    height=200,
    placeholder="Today I feel..."
)

# ─── 4) Run inference & display ───────────────────────────────────────────────
if st.button("🔮 Predict Mood"):
    if not journal_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Tokenize + forward
        inputs = tokenizer(
            journal_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits  # shape (1, num_labels)
        probs = torch.sigmoid(logits)[0].cpu().tolist()

        # Map label‐id → label name
        id2label = model.config.id2label  # e.g. {0: 'admiration', 1: 'amusement', ...}
        pairs = [(id2label[i], probs[i]) for i in range(len(probs))]
        # Sort by probability desc and take top‐5
        top5 = sorted(pairs, key=lambda x: x[1], reverse=True)[:5]

        # Show as a table
        table = {label: f"{score:.3f}" for label, score in top5}
        st.success("📝 Top 5 Emotions")
        st.table(table)

        # Optional “flair” based on top‐1
        top1 = top5[0][0]
        if top1 == "joy":
            st.balloons()
            st.info("💛 Keep that joy alive!")
        elif top1 in ("sadness", "disappointment", "grief"):
            st.warning("💙 It's okay to feel down. Take care of yourself.")
        elif top1 in ("anger", "annoyance", "disgust"):
            st.error("😡 I sense some anger. Maybe take a few deep breaths?")
        elif top1 in ("fear", "nervousness"):
            st.info("🧘‍♂️ You’ve got this. Breathe in… breathe out…")
        else:
            st.info("🌤️ Feeling complex emotions – that's perfectly normal.")

# ─── 5) Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("app/avatar.jpg", width=100)
    st.markdown("### 🤖 MoodSense")
    st.markdown("Built by Tu – I love AI")
    st.markdown("[GitHub Repo](https://github.com/salaghati/MoodSense)")
    st.markdown("— Powered by Streamlit & Transformers —")
