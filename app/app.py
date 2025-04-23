import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# 1) Page config
st.set_page_config(
    page_title="MoodSense by Tu",
    page_icon="🧠",
    layout="centered"
)

# 1) Lấy token từ secrets
hf_token = st.secrets["HF_TOKEN"]

# 2) Repo HF của bạn
REPO_ID = "Salaghati/moodsense-distilbert"

# 3) Load với auth token để tránh rate‐limit
tokenizer = AutoTokenizer.from_pretrained(
    REPO_ID,
    use_auth_token=hf_token
)
model = AutoModelForSequenceClassification.from_pretrained(
    REPO_ID,
    problem_type="multi_label_classification",
    use_auth_token=hf_token
)
model.eval()

# 3) Build a pipeline for multi-label classification
#    On Mac MPS use device=0, otherwise CPU device=-1
device = 0 if torch.backends.mps.is_available() else -1
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device,
    return_all_scores=True
)

# 4) Header
st.image("https://i.imgur.com/QzloMmO.png", width=120)
st.markdown("## 👋 Hi! How are you feeling today?")

# 5) User input
journal_text = st.text_area(
    "✍️ Enter your journal entry below:",
    height=200,
    placeholder="Today I feel..."
)

# 6) Predict button
if st.button("🔮 Predict Mood"):
    if journal_text.strip():
        # Run inference
        outputs = classifier(journal_text, top_k=5)[0]
        # Display top-5 scores
        scores = {item["label"]: f"{item['score']:.3f}" for item in outputs}
        st.success("📝 Top Predictions:")
        st.table(scores)

        # Custom feedback based on top label
        top1 = outputs[0]["label"].lower()
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
    else:
        st.warning("Please enter some text to analyze.")

# 7) Sidebar
with st.sidebar:
    st.image("app/avatar.jpg", width=100)
    st.markdown("### 🤖 MoodSense")
    st.markdown("Built by Tu – I love AI")
    st.markdown("[GitHub Repo](https://github.com/salaghat1/MoodSense)")
    st.markdown("— Powered by Streamlit & Transformers —")
