import streamlit as st
import joblib
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AutoJudge",
    page_icon="⚖️",
    layout="wide"
)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    # Silent NLTK downloads
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
    try:
        # Load all models and scalers
        # Ensure these are the NEW models trained on 1-10 data
        clf = joblib.load('models/voting_classifier.pkl')
        reg = joblib.load('models/gb_regressor.pkl')
        tfidf = joblib.load('models/tfidf.pkl')
        scaler = joblib.load('models/scaler.pkl')
        le = joblib.load('models/label_encoder.pkl')
        return clf, reg, tfidf, scaler, le
    except FileNotFoundError:
        return None, None, None, None, None

voting_clf, gb_reg, tfidf_vectorizer, scaler, label_encoder = load_resources()

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- PREPROCESSING FUNCTIONS (MATCHING NOTEBOOK) ---
def expert_cleaning(text):
    if not isinstance(text, str): return ""
    
    # 1. Capture Constraints
    text = re.sub(r'10\^\{?[5-9]\}?', ' heavy_constraint ', text) 
    text = re.sub(r'10\^\{?1[0-8]\}?', ' heavy_constraint ', text)
    text = re.sub(r'\d+(\.\d+)?\s?seconds?', ' time_limit ', text)
    
    # 2. Clean & Tokenize
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s\%\^\<\>\=\_]', ' ', text)
    
    tokens = text.split()
    
    # 3. Lemmatize
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)

def get_handcrafted_features(text):
    # Topic Counters
    dp = len(re.findall(r'dynamic programming|optimal|maximize|minimize|subsequence', text))
    graph = len(re.findall(r'graph|vertex|edge|tree|ancestor|connected component', text))
    math = len(re.findall(r'modulo|prime|gcd|divisible|remainder', text))
    geo = len(re.findall(r'geometry|convex|polygon|angle', text))
    brute = len(re.findall(r'iterate|every possible|check all', text))
    
    # Structure Counters
    length = len(text.split())
    math_symbols = len(re.findall(r'[\%\^\<\>\=]', text))
    
    return [dp, graph, math, geo, brute, length, math_symbols]

# --- UI LAYOUT ---
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0px;'>AutoJudge</h1>
    <h3 style='text-align: center; color: grey;'>Programming Problem Difficulty Predictor</h3>
    <hr>
    """, 
    unsafe_allow_html=True
)

# Check if models loaded
if not voting_clf:
    st.error("⚠️ Models not found! Please re-run 'Autojudge_main.ipynb' to generate the 'models/' folder.")
    st.stop()

# Input Columns
col_left, col_right = st.columns([1.5, 1], gap="large")

with col_left:
    st.markdown("### Problem Statement")
    
    desc = st.text_area(
        "Problem Description", 
        height=200, 
        placeholder="Paste the problem story here..."
    )
    
    c1, c2 = st.columns(2)
    with c1:
        input_desc = st.text_area("Input Format", height=100, placeholder="e.g. First line contains T...")
    with c2:
        output_desc = st.text_area("Output Format", height=100, placeholder="e.g. Print the answer...")

with col_right:
    st.markdown("### Prediction")
    st.write("Analyze the complexity of the problem.")
    
    if st.button(" Analyze Difficulty", type="primary", use_container_width=True):
        if not desc:
            st.warning("Please provide a description.")
        else:
            with st.spinner("Crunching numbers..."):
                # A. Prepare Text
                full_text = f"{desc} {input_desc} {output_desc}"
                clean_text = expert_cleaning(full_text)
                
                # B. Extract Features
                X_tfidf = tfidf_vectorizer.transform([clean_text])
                handcrafted_raw = get_handcrafted_features(clean_text)
                
                # C. Proxy Rarity & Stack
                rarity_val = X_tfidf.sum()
                dense_vector = np.array([handcrafted_raw + [rarity_val]])
                dense_scaled = scaler.transform(dense_vector)
                
                # D. Final Matrix
                X_final = hstack([X_tfidf, csr_matrix(dense_scaled)])
                
                # E. Predict
                pred_class_idx = voting_clf.predict(X_final)[0]
                pred_class = label_encoder.inverse_transform([pred_class_idx])[0]
                
                # Predict Score (Now returns 1-10 because we retrained it)
                pred_score = gb_reg.predict(X_final)[0]
                
                # F. Display Results
                st.markdown("---")
                
                # 1. Class Display
                st.info(f"**Class**\n# {pred_class}")

                # 2. Score Display (1-10 Scale)
                st.info(f"**Difficulty Score**\n# {pred_score:.1f} / 10")
                
                # 3. Progress Bar (Normalized 0-10 -> 0.0-1.0)
                st.caption("Difficulty Meter")
                bar_val = min(max(pred_score / 10.0, 0.0), 1.0)
                st.progress(bar_val)