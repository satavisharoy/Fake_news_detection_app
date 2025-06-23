import streamlit as st
import joblib

st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e8c3fd, #ffd1dc);
        background-attachment: fixed;
    }

    .block-container {
        background-color: rgba(255, 255, 255, 0.88);
        padding: 2rem;
        border-radius: 12px;
        max-width: 700px;
        margin: auto;
        box-shadow: 0px 0px 25px rgba(0, 0, 0, 0.05);
    }

    h1, p {
        text-align: center;
        color: #2c2c2c;
    }

    .stTextArea textarea {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }

    .stButton>button {
        background-color: #9b59b6 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 1.5em;
        font-size: 16px;
    }

    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>📰 Fake News Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p>Enter a news article below to check whether it's <b>Real</b> or <b>Fake</b>.</p>", unsafe_allow_html=True)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

user_input = st.text_area("📝 Paste the News Article Here:", height=200)

if st.button("🚀 Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text to check.")
    else:
        input_transformed = vectorizer.transform([user_input])
        prediction = model.predict(input_transformed)

        if prediction[0] == 1:
            st.success("✅ This news article is REAL.")
        else:
            st.error("🚨 This news article is FAKE.")