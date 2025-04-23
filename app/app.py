import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── 0) Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoodSense by Tu",
    page_icon="🧠",
    layout="centered"
)
st.image("https://i.imgur.com/QzloMmO.png", width=120)
st.markdown("## 👋 Hi! How are you feeling today?")

# ─── 1) Load HF model & tokenizer ───────────────────────────────────────────────
hf_token = st.secrets["HF_TOKEN"]
repo_id  = "Salaghati/moodsense-distilbert"

tokenizer = AutoTokenizer.from_pretrained(
    repo_id, use_auth_token=hf_token
)
model = AutoModelForSequenceClassification.from_pretrained(
    repo_id,
    problem_type="multi_label_classification",
    use_auth_token=hf_token
)
model.eval()

# ─── 2) Device setup ─────────────────────────────────────────────────────────────
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# ─── 3) User input ────────────────────────────────────────────────────────────────
journal_text = st.text_area(
    "✍️ Enter your journal entry below:",
    height=200,
    placeholder="Today I feel..."
)

# ─── 4) Inference & display ───────────────────────────────────────────────────────
if st.button("🔮 Predict Mood"):
    if not journal_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Tokenize (returns dict of tensors, NOT a single tensor)
        enc = tokenizer(
            journal_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )
        # Move tensors to device
        enc = {k: v.to(device) for k, v in enc.items()}

        # Forward pass
        with torch.no_grad():
            logits = model(**enc).logits  # shape (1, num_labels)

        # Convert to probabilities
        probs = torch.sigmoid(logits)[0].cpu().tolist()

        # Map ids to labels and sort
        id2label = model.config.id2label
        pairs = [(id2label[i], probs[i]) for i in range(len(probs))]
        top5  = sorted(pairs, key=lambda x: x[1], reverse=True)[:5]

        # Display as table
        table = {lbl: f"{score:.3f}" for lbl, score in top5}
        st.success("📝 Top 5 Emotions")
        st.table(table)

        # Custom messages on top-1
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

# ─── 5) Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("app/avatar.jpg", width=100)
    st.markdown("### 🤖 MoodSense")
    st.markdown("Built by Tu – I love AI")
    st.markdown("[GitHub Repo](https://github.com/salaghat1/MoodSense)")
    st.markdown("— Powered by Streamlit & Transformers —")
