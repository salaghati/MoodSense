import streamlit as st
import joblib
import os

st.set_page_config(page_title='MoodSense', layout='centered')
st.title("🧠 MoodSense - Journal Mood Predictor")

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
if os.path.exists(model_path):
    model = joblib.load(model_path)
    journal_text = st.text_area("Enter your journal entry:", height=200)

    if st.button("Predict Mood"):
        if journal_text.strip():
            prediction = model.predict([journal_text])[0]
            st.success(f"📝 Predicted Mood: **{prediction}**")
            if prediction == "Sad":
                st.info("Try some self-care today. A walk or a chat might help. 💙")
            elif prediction == "Anxious":
                st.info("Take a deep breath. You're doing your best. 🧘")
            elif prediction == "Happy":
                st.balloons()
                st.info("Keep it up! Spread that joy! 🌞")
            else:
                st.info("A calm day is a good day. Stay steady. ⚖️")
        else:
            st.warning("Please enter some text first.")
else:
    st.error("Model file not found. Please train the model first.")
