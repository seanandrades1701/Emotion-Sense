
import re
import joblib
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from scipy.sparse import hstack

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Emotion Sense | Marathi NLP",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# EMOTION CONFIGURATION
# =========================================================

EMOTIONS = {
    "आनंदी": {
        "english": "Happy",
        "description": "Positive feelings, happiness and satisfaction."
    },
    "आश्चर्य": {
        "english": "Surprise",
        "description": "Unexpected events, amazement and astonishment."
    },
    "दुःखी": {
        "english": "Sad",
        "description": "Sadness, disappointment and emotional distress."
    },
    "भीती": {
        "english": "Fear",
        "description": "Fear, anxiety and feelings of uncertainty."
    },
    "राग": {
        "english": "Angry",
        "description": "Anger, frustration and dissatisfaction."
    },
    "सामान्य": {
        "english": "Neutral",

        "description": "Factual or emotionally neutral language."
    }
}

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
/* ---------- GLOBAL ---------- */



#MainMenu,
footer {
    visibility: hidden;
}

.block-container {
    max-width: 1120px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

html, body, [class*="css"] {
    font-family: "Nirmala UI", "Segoe UI", sans-serif;
}

p, label, li {
    color: #b9c9bf;
}

/* ---------- HERO ---------- */

.hero {
    position: relative;
    overflow: hidden;
    padding: 48px 40px 42px;
    margin: 8px 0 28px;
    border: 1px solid rgba(74, 222, 128, 0.20);
    border-radius: 26px;
    background: linear-gradient(
        130deg,
        rgba(22, 49, 34, 0.95),
        rgba(12, 23, 17, 0.90)
    );
    box-shadow: 0 18px 60px rgba(0, 0, 0, 0.20);
}

.hero::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -80px;
    top: -120px;
    border-radius: 50%;
    background: rgba(74, 222, 128, 0.12);
    filter: blur(3px);
}

.eyebrow {
    color: #86efac;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: 11px;
    font-weight: 800;
    margin-bottom: 15px;
}

.hero-title {
    font-size: clamp(38px, 5vw, 62px);
    line-height: 1.12;
    letter-spacing: -2px;
    font-weight: 850;
    color: #f0fdf4;
    margin: 0;
}

.hero-title .accent {
    background: linear-gradient(90deg, #4ade80, #a3e635);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero-description {
    max-width: 660px;
    color: #b6c9bb;
    font-size: 15px;
    line-height: 1.85;
    margin-top: 17px;
    margin-bottom: 0;
}

.hero-badge {
    display: inline-block;
    margin-top: 23px;
    padding: 8px 13px;
    border: 1px solid rgba(74, 222, 128, 0.26);
    border-radius: 100px;
    color: #bbf7d0;
    background: rgba(34, 197, 94, 0.09);
    font-size: 12px;
    font-weight: 600;
}

/* ---------- SECTION HEADINGS ---------- */

.section-label {
    color: #4ade80;
    font-size: 11px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    font-weight: 800;
    margin-bottom: 8px;
}

.section-title {
    color: #f0fdf4;
    font-size: 25px;
    font-weight: 750;
    margin: 0 0 7px;
    letter-spacing: -0.5px;
}

.section-description {
    color: #92a89a;
    font-size: 13px;
    line-height: 1.7;
    margin-bottom: 20px;
}

/* ---------- PANELS ---------- */

.panel {
    padding: 24px;
    border: 1px solid rgba(150, 190, 162, 0.15);
    border-radius: 20px;
    background: linear-gradient(
        145deg,
        rgba(22, 35, 27, 0.96),
        rgba(13, 22, 16, 0.96)
    );
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.12);
}

.panel-title {
    color: #e7f5eb;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 6px;
}

.panel-caption {
    color: #8fa696;
    font-size: 12px;
    margin-bottom: 18px;
}

/* ---------- TEXT AREA ---------- */

textarea {
    background: #0b140e !important;
    color: #f0fdf4 !important;
    border: 1px solid #294a34 !important;
    border-radius: 13px !important;
    font-family: "Nirmala UI", "Segoe UI", sans-serif !important;
    font-size: 17px !important;
    line-height: 1.9 !important;
}

textarea:focus {
    border-color: #4ade80 !important;
    box-shadow: 0 0 0 1px #4ade80 !important;
}

[data-testid="stTextArea"] label {
    color: #d3e5d8 !important;
    font-weight: 650 !important;
}


/* ---------- BUTTONS ---------- */

.stButton > button {
    width: 100%;
    min-height: 49px;
    border-radius: 12px;
    border: 1px solid #4ade80;
    background: linear-gradient(100deg, #22c55e, #86efac);
    color: #07140b;
    font-size: 14px;
    font-weight: 800;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(100deg, #4ade80, #bef264);
    border-color: #86efac;
    color: #07140b;
    box-shadow: 0 5px 24px rgba(74, 222, 128, 0.18);
    transform: translateY(-1px);
}

.stButton > button:focus {
    color: #07140b;
}

/* Only the Analyze emotion button */
div[data-testid="stButton"]:has(button[kind="primary"]) button {
    color: #000000 !important;
}

div[data-testid="stButton"]:has(button[kind="primary"]) button p {
    color: #000000 !important;
}


/* ---------- RESULT ---------- */

.result-card {
    padding: 30px 22px;
    text-align: center;
    border-radius: 21px;
    border: 1px solid rgba(74, 222, 128, 0.28);
    background:
        radial-gradient(circle at 50% 0%, rgba(74, 222, 128, 0.12), transparent 60%),
        linear-gradient(145deg, #14291b, #0d1811);
}

.result-overline {
    color: #8eb99b;
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 2.2px;
    font-weight: 800;
}

.result-emotion {
    font-size: 34px;
    font-weight: 850;
    color: #86efac;
    line-height: 1.5;
}

.result-english {
    color: #d0dfd4;
    font-size: 15px;
    margin-top: 1px;
}

.result-description {
    color: #91a99a;
    font-size: 12px;
    line-height: 1.7;
    margin-top: 13px;
}

.metric-card {
    border: 1px solid rgba(150, 190, 162, 0.15);
    border-radius: 14px;
    padding: 15px 17px;
    background: rgba(19, 32, 23, 0.85);
}

.metric-label {
    color: #91a99a;
    font-size: 11px;
    margin-bottom: 7px;
}

.metric-value {
    color: #e8f8ed;
    font-size: 24px;
    font-weight: 800;
}

/* ---------- DIVIDER & FOOTER ---------- */

hr {
    border-color: rgba(150, 190, 162, 0.14) !important;
    margin-top: 28px !important;
    margin-bottom: 28px !important;
}

.footer {
    padding-top: 15px;
    text-align: center;
    color: #718678;
    font-size: 11px;
    line-height: 1.9;
}

.footer strong {
    color: #4ade80;
}

/* ---------- RESPONSIVE ---------- */

@media (max-width: 640px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }

    .hero {
        padding: 30px 23px;
        border-radius: 19px;
    }

    .hero-title {
        letter-spacing: -1px;
    }

    .panel {
        padding: 18px;
    }

    .result-emotion {
        font-size: 28px;
    }
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_resources():
    with open("stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = {
            word.strip()
            for word in file
            if word.strip()
        }

    word_vectorizer = joblib.load(
        "models/word_tfidf.pkl"
    )

    char_vectorizer = joblib.load(
        "models/char_tfidf.pkl"
    )

    model = joblib.load(
        "models/emotion_model_word_char.pkl"
    )

    return stopwords, word_vectorizer, char_vectorizer, model


# =========================================================
# PREPROCESSING
# =========================================================

def clean_text(text):
    text = str(text)

    # Preserve Devanagari characters and matras.
    text = re.sub(
        r"[^\u0900-\u097F\sA-Za-z]",
        "",
        text
    )

    text = re.sub(
        r"[0-9०-९]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def preprocess_text(text, stopwords):
    cleaned = clean_text(text)
    tokens = cleaned.split()

    tokens = [
        word for word in tokens
        if word not in stopwords
    ]

    return " ".join(tokens)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">Natural Language Processing · Marathi</div>
    <div class="hero-title">
        Emotion <span class="accent">Sense.</span>
    </div>
    <p class="hero-description">
        Understand the emotions behind Marathi text.
        Analyze a sentence using Natural Language Processing,
        TF-IDF feature extraction and machine learning.
    </p>
    <div class="hero-badge">
        ● &nbsp; WORD + CHARACTER TF-IDF &nbsp; · &nbsp; LOGISTIC REGRESSION
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# INPUT SECTION
# =========================================================

st.markdown("""
<div class="section-label">01 / Text Analysis</div>
<div class="section-title">What does your text express?</div>
<div class="section-description">
    Enter a Marathi sentence and let the model predict its emotional category.
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.6, 1], gap="large")

with left:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Marathi text input</div>
        <div class="panel-caption">
            Write or paste your sentence in Devanagari script.
        </div>
    """, unsafe_allow_html=True)

    sentence = st.text_area(
        "Marathi sentence",
        placeholder="उदा. आज मला खूप आनंद झाला.",
        height=160,
        max_chars=2000,
        label_visibility="collapsed",
        key="marathi_input"
    )

    st.caption(
        f"{len(sentence)} / 2000 characters"
    )

    analyze = st.button(
        "Analyze emotion",
        type="primary",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Six emotion categories</div>
        <div class="panel-caption">
            The classifier predicts one of these labels.
        </div>
    """, unsafe_allow_html=True)

    for marathi, info in EMOTIONS.items():
        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:11px;
                padding:9px 0;
                border-bottom:1px solid rgba(150,190,162,0.10);
            ">
                <div style="font-size:22px;width:30px;">
                </div>
                <div style="flex:1;">
                    <div style="color:#e7f5eb;font-size:14px;font-weight:700;">
                        {marathi}
                    </div>
                    <div style="color:#829889;font-size:11px;">
                        {info['english']}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PREDICTION
# =========================================================

if analyze:
    if not sentence.strip():
        st.warning("कृपया विश्लेषणासाठी मराठी वाक्य लिहा.")

    else:
        try:
            (
                stopwords,
                word_vectorizer,
                char_vectorizer,
                model
            ) = load_resources()

            processed = preprocess_text(
                sentence,
                stopwords
            )

            if not processed:
                st.warning(
                    "Preprocessing नंतर वापरण्यायोग्य मजकूर उरलेला नाही."
                )

            else:
                with st.spinner("Analyzing Marathi text..."):
                    word_features = word_vectorizer.transform(
                        [processed]
                    )

                    char_features = char_vectorizer.transform(
                        [processed]
                    )

                    combined_features = hstack([
                        word_features,
                        char_features
                    ])

                    prediction = model.predict(
                        combined_features
                    )[0]

                    probabilities = model.predict_proba(
                        combined_features
                    )[0]

                    classes = list(model.classes_)

                    probability_map = {
                        classes[i]: float(probabilities[i] * 100)
                        for i in range(len(classes))
                    }

                    top_probability = probability_map[prediction]
                    info = EMOTIONS.get(
                        prediction,
                        {
                            "english": "Unknown",
                            "description": "Predicted category."
                        }
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown("""
                <div class="section-label">02 / Analysis Result</div>
                <div class="section-title">Your emotion analysis</div>
                <div class="section-description">
                    The result below is generated by your trained machine-learning model.
                </div>
                """, unsafe_allow_html=True)

                result_left, result_right = st.columns(
                    [1, 1.15],
                    gap="large"
                )

                with result_left:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-overline">
                                Predicted emotion
                            </div>
                            <div class="result-emotion">
                                {prediction}
                            </div>
                            <div class="result-english">
                                {info['english']}
                            </div>
                            <div class="result-description">
                                {info['description']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with result_right:
                    st.markdown("""
                    <div class="panel">
                        <div class="panel-title">Prediction details</div>
                        <div class="panel-caption">
                            Model-estimated class probabilities.
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                Predicted class probability
                            </div>
                            <div class="metric-value">
                                {top_probability:.2f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown(
                        "**Probability by emotion**"
                    )

                    for emotion, score in sorted(
                        probability_map.items(),
                        key=lambda item: item[1],
                        reverse=True
                    ):
                        display_name = EMOTIONS.get(
                            emotion,
                            {}
                        ).get("english", emotion)

                        st.markdown(
                            f"""
                            <div style="
                                display:flex;
                                justify-content:space-between;
                                gap:12px;
                                margin-top:12px;
                                margin-bottom:4px;
                                color:#cbded0;
                                font-size:12px;
                            ">
                                <span>{emotion} · {display_name}</span>
                                <span>{score:.1f}%</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.progress(
                            min(max(score / 100, 0.0), 1.0)
                        )

                    st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("View text preprocessing"):
                    st.markdown("**Original sentence**")
                    st.write(sentence)

                    st.markdown("**Processed sentence**")
                    st.code(
                        processed,
                        language=None
                    )

                st.caption(
                    "Note: These are model-estimated probabilities, "
                    "not guarantees of correctness. Emotion classification "
                    "depends on context and wording."
                )

        except FileNotFoundError as error:
            st.error(
                f"Required file not found: {error}. "
                "Check your models folder and stopwords.txt."
            )

        except Exception as error:
            st.error(
                f"Unable to analyze the sentence: {error}"
            )

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<hr>
<div class="footer">
    <strong>EMOTION SENSE</strong><br>
    Marathi Emotion Classification · NLP Mini Project<br>
    Word TF-IDF + Character TF-IDF · Logistic Regression
</div>
""", unsafe_allow_html=True)
