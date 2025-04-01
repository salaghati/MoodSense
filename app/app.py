import streamlit as st
import joblib
import os

st.set_page_config(page_title="MoodSense by Tu", page_icon="🧠", layout="centered")

st.markdown(
    '''
    <div style="text-align: center; margin-bottom: 10px;">
        <img src="app/avatar.jpg" alt="Avatar" style="border-radius: 50%; width: 120px; border: 3px solid #aaa;" />
        <h2 style="margin-top: 10px;">👋 Hi Tu! How are you feeling today?</h2>
    </div>
    ''',
    unsafe_allow_html=True
)

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
    st.markdown("Built by Tu – An AI Engineer in the making 💻")
    st.markdown("[GitHub Repo](https://github.com/salaghat1/MoodSense)")
    st.markdown("— Powered by Streamlit & scikit-learn —")
