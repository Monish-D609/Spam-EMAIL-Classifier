# 0 - Spam, 1 - Ham

import streamlit as st
import pandas as pd
import joblib 
model = joblib.load("model.pkl")
threshold = joblib.load("threshold.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("EMAIL Spam Classifier")
st.write("Enter your raw email text here to predict SPAM or HAM")

input_text = st.text_area(
    "Enter your email here:",
    height=150,
    placeholder="Type or paste the email content..."
)

input = [input_text]



# Function the prints the output based on user input
def predict_spam_ham(user_input: str):
    # Vectorizer expects a list of strings
    input_feature = vectorizer.transform([user_input])

    spam_prob = model.predict_proba(input_feature)[0, 1]

    final_prediction = int(spam_prob >= threshold)

    return final_prediction, spam_prob


if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter an email.")
    else:
        pred, prob = predict_spam_ham(input_text)

        st.write(f"**Spam Probability:** {prob:.2f}")
        st.write(f"**Threshold:** {threshold:.2f}")

        if pred == 0:
            st.error("🚨 **SPAM** — This email is likely spam.")
        else:
            st.success("✅ **HAM** — This email is likely safe.")