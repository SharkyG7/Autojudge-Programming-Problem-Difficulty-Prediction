#  AutoJudge: AI-Powered Problem Difficulty Predictor

**AutoJudge** is an intelligent machine learning system that predicts the difficulty of competitive programming problems based **solely on their textual description**. 

Instead of relying on heavy Deep Learning models (like BERT), this project demonstrates the power of **Classical ML combined with Expert Feature Engineering** to categorize problems as **Easy, Medium, or Hard** and predict a precise **1-10 Difficulty Score**.

---

##  Key Features

* **Dual Prediction:** Predicts both a **Category** (Classification) and a precise **Numerical Score** (Regression).
* **Expert Text Cleaning:** Custom regex pipeline to detect constraints (e.g., $N \le 10^9$) which are critical indicators of algorithmic complexity.
* **Cognitive Features:** Extracts "human-like" features such as:
    * **Math Density:** Frequency of symbols like `%`, `^`, `|`.
    * **Topic Detection:** Keywords for DP, Graphs, Geometry, etc.
    * **Constraint Intensity:** Time limits and input sizes.
* **Interactive Web App:** A clean, user-friendly interface built with Streamlit.

---

##  Tech Stack

* **Language:** Python
* **Machine Learning:** Scikit-Learn (Voting Classifier, Gradient Boosting, TF-IDF)
* **NLP:** NLTK (Lemmatization, Stopwords)
* **Web Framework:** Streamlit
* **Data Handling:** Pandas, NumPy, Scipy (Sparse Matrices)

---

##  Project Structure

```text
📁 AutoJudge/
├── 📄 Autojudge_main.ipynb    # Main Jupyter Notebook (Training & Evaluation)
├── 📄 streamlit_app.py        # The Web Application
├── 📄 problems_data.jsonl     # Dataset (Input Data)
├── 📄 README.md               # Project Documentation
└── 📁 models/                 # Generated after running the notebook
    ├── voting_classifier.pkl
    ├── gb_regressor.pkl
    ├── tfidf.pkl
    ├── scaler.pkl
    └── label_encoder.pkl

```

---

##  Installation & Usage

Follow these steps to set up the project locally.

### 1. Clone the Repository

```bash
git clone [https://github.com/yourusername/AutoJudge.git](https://github.com/yourusername/AutoJudge.git)
cd AutoJudge

```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn nltk streamlit joblib scipy matplotlib seaborn

```

### 3. Train the Models

**Crucial Step:** You must run the training notebook first to generate the model files.

1. Open `Autojudge_main.ipynb` in Jupyter Notebook or VS Code.
2. **Run All Cells**.
3. Wait for the final cell to confirm: `✅ All files saved to 'models/' directory.`

### 4. Run the Web App

Once the `models/` folder is populated, launch the application:

```bash
streamlit run streamlit_app.py

```

A browser window will open automatically at `http://localhost:8501`.

---

##  Methodology Highlights

### 1. The "Expert Cleaning" Pipeline

Standard text cleaning removes numbers and symbols, but for programming problems,  is a massive clue.

* **We convert:** `$10^9$`  `heavy_constraint`
* **We keep:** `%`, `^`, `<`, `>` (Math symbols)
* **We lemmatize:** `running`  `run`

### 2. Hybrid Feature Matrix

We combine two types of data for the model:

* **Sparse Vector:** 5,000 TF-IDF features (What the problem is *about*).
* **Dense Vector:** 7 Handcrafted features (How *complex* the text structure is).

### 3. Ensemble Learning

* **Classifier:** A **Voting Ensemble** (Logistic Regression + Random Forest) captures both linear patterns and non-linear decision trees.
* **Regressor:** A **Gradient Boosting Regressor** minimizes error for the precise 1-10 score.

---

## 🎥 Project Demo Video

Watch the system in action here:
**[▶️ Click Here to Watch the Demo Video]**
https://drive.google.com/file/d/1qqWzhw4QloOjb8Ne7WpVUrkf8EYD6LIc/view?usp=drive_link

---

## 📸 Screenshots

### 1. The Web Interface

<img width="1798" height="644" alt="image" src="https://github.com/user-attachments/assets/e71f90f8-81c2-4daa-b27b-3dfa801f3af8" />
<img width="1782" height="754" alt="image" src="https://github.com/user-attachments/assets/c1df20bc-2bde-43e2-b495-d1bcf6be8df3" />


### 2. Confusion Matrix (Results)

<img width="663" height="566" alt="image" src="https://github.com/user-attachments/assets/830fba78-249d-44df-8baa-6e2aca370774" />


---

##  Author

**Gaurav Arora (23322013), BS-MS Economics, IIT Roorkee**
