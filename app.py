import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="centered"
)

# ---------------- LIGHT UI STYLE ----------------
st.markdown("""
<style>
.stApp { background-color: #f5f7fa; font-family: 'Segoe UI', sans-serif; }
.main-card { background-color: white; padding: 40px; border-radius: 15px; box-shadow: 0px 4px 15px rgba(0,0,0,0.08); }
.title { font-size: 34px; font-weight: 600; color: #2c3e50; text-align: center; }
.subtitle { text-align: center; color: #7f8c8d; margin-bottom: 25px; }
.result-box { background-color: #eaf4ff; color: #1f4e79; padding: 20px; border-radius: 10px; font-size: 22px; font-weight: 500; text-align: center; margin-top: 20px; }
.stButton>button { background-color: #4a90e2; color: white; border-radius: 8px; padding: 8px 25px; font-size: 16px; border: none; }
.stButton>button:hover { background-color: #357abd; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
try:
    svm_model = pickle.load(open("svm_spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
except:
    st.error("Model files not found! Place 'svm_spam_model.pkl' and 'tfidf_vectorizer.pkl' in this folder.")
    st.stop()

# ---------------- UI ----------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="title">📧 Spam Mail Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your email content below to detect Spam</div>', unsafe_allow_html=True)

email_text = st.text_area("Enter Email Content Here:", height=150)

if st.button("Predict Spam"):
    if email_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Vectorize input
        X_input = vectorizer.transform([email_text])
        
        # Predict class
        prediction = svm_model.predict(X_input)[0]  # 1 = Spam, 0 = Not Spam
        decision_score = svm_model.decision_function(X_input)[0]

        result_text = "Spam" if prediction == 1 else "Not Spam"

        # ---------- Confidence using Sigmoid ----------
        prob = 1 / (1 + np.exp(-decision_score))
        # Invert if predicted class is 0 (Not Spam)
        if prediction == 0:
            prob = 1 - prob

        confidence = prob * 100

        # ---------- Result Box ----------
        st.markdown(f"""
        <div class="result-box">
            Prediction: {result_text} <br>
            Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

        # ---------- Probability Bars ----------
        if prediction == 1:  # Spam
            spam_prob = int(confidence)
            not_spam_prob = 100 - spam_prob
        else:  # Not Spam
            not_spam_prob = int(confidence)
            spam_prob = 100 - not_spam_prob

        st.subheader("Probability Estimate")
        st.write("Spam")
        st.progress(spam_prob)
        st.write("Not Spam")
        st.progress(not_spam_prob)

st.markdown('</div>', unsafe_allow_html=True)