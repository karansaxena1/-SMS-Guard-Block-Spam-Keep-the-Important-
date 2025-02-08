import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download the stopwords resource
nltk.download('stopwords')
nltk.download('punkt_tab')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorize2.pkl', 'rb'))
model = pickle.load(open('model2.pkl', 'rb'))

st.set_page_config(page_title="Spam/Ham Identifier")
st.title("📲🚫 SMS Guard: Block Spam, Keep the Important!🔍")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("🚨📩 It's a SPAM!! 🛑 Don't Fall for Fake Texts! 🔍")
    else:
        st.header("📩✅ Not Spam! This Message is Safe & Important! 📬💙")
