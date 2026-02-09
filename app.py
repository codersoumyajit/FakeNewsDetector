import streamlit as st
import joblib

# ---------------- Load Model ----------------
@st.cache_resource
def load_models():
    return joblib.load("vectorizer.jb"), joblib.load("lr_model.jb")

vectorizer, model = load_models()

# ---------------- Page Config ----------------
st.set_page_config(page_title="Fake News AI", page_icon="🧠", layout="centered")

# ---------------- Custom CSS ----------------
st.markdown("""
<style>

/* Background Gradient */
body {
    background: linear-gradient(135deg, #0f172a, #0b2447, #0a2647);
}


/* Main container glass effect */
.main {
    background: #eef2ff;
    backdrop-filter: blur(20px);
    border-radius: 25px;
    padding: 20px;
}

/* Title styling */
h1 {
    text-align: center;
    font-size: 3rem !important;
    background: linear-gradient(90deg, #00dbde, #fc00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Text Area */
textarea {
    background: #eef2ff !important;
    color: #111827 !important;
    border: 1px solid #6366f1 !important;
}

/* Button styling */
button {
    background: linear-gradient(90deg, #7c3aed, #06b6d4) !important;
    color: white !important;
    font-size: 18px !important;
    border-radius: 12px !important;
    padding: 10px 30px !important;
    border: none !important;
    transition: 0.3s ease-in-out;
}

button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 20px #7c3aed;
}

/* Result box */
.result-box {
    font-size: 22px;
    font-weight: bold;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}

.real {
    background: rgba(34,197,94,0.2);
    border: 1px solid #22c55e;
    color: #22c55e;
}

.fake {
    background: rgba(239,68,68,0.2);
    border: 1px solid #ef4444;
    color: #ef4444;
}

</style>
""", unsafe_allow_html=True)

# ---------------- UI ----------------
st.markdown("<h1>🧠 Fake News AI Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94a3b8;'>Paste any news article and AI will verify authenticity</p>", unsafe_allow_html=True)

news_input = st.text_area("📰 News Article", height=180)

# ---------------- Prediction ----------------
if st.button("🚀 Check News"):
    if news_input.strip():
        transformed = vectorizer.transform([news_input])
        prediction = model.predict(transformed)

        if prediction[0] == 1:
            st.markdown("<div class='result-box real'>✅ This News is REAL</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box fake'>❌ This News is FAKE</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("<p style='text-align:center;color:#64748b;margin-top:30px;'>Made with ❤️ using Machine Learning</p>", unsafe_allow_html=True)
