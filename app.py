import streamlit as st
import joblib

# Load model + vectorizer
model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Map emotions to emojis and colors
emotion_style = {
    "joy": ("😀", "yellow"),
    "sadness": ("😢", "blue"),
    "anger": ("😡", "red"),
    "fear": ("😨", "purple"),
    "love": ("❤️", "pink"),
    "surprise": ("😲", "orange")
}

# Show what model can predict
st.write("Emotions model can predict:", model.classes_)

st.title("Real-Time Emotion Detection from Text")
user_input = st.text_area("Enter your text here:")

if st.button("Predict Emotion"):
    if user_input.strip() != "":
        X = vectorizer.transform([user_input])
        pred = model.predict(X)[0]

        emoji, color = emotion_style.get(pred, ("❓", "gray"))
        st.markdown(f"<h2 style='color:{color}'>{emoji} {pred}</h2>", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text.")
