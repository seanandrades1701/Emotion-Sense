import pandas as pd
import re
import sys
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

sys.stdout.reconfigure(encoding="utf-8")

print("=" * 70)
print("UNSEEN MARATHI EMOTION DATASET EVALUATION")
print("WORD + CHARACTER TF-IDF MODEL")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD UNSEEN DATASET
# ---------------------------------------------------------

df = pd.read_csv("dataset/unseen_test_300.csv")
df.columns = df.columns.str.strip()

print("\nDataset Shape:", df.shape)

print("\nClass Distribution:")
print(df["Emotion"].value_counts())

# ---------------------------------------------------------
# 2. LOAD STOPWORDS
# ---------------------------------------------------------

with open("stopwords.txt", "r", encoding="utf-8") as file:
    stopwords = set(
        word.strip()
        for word in file
        if word.strip()
    )

# ---------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------

def clean_text(text):
    text = str(text)

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


def preprocess_text(text):
    text = clean_text(text)

    tokens = text.split()

    tokens = [
        word
        for word in tokens
        if word not in stopwords
    ]

    return " ".join(tokens)


df["Processed_Sentence"] = df["Sentence"].apply(
    preprocess_text
)

# ---------------------------------------------------------
# 4. LOAD TRAINED VECTORIZERS + MODEL
# ---------------------------------------------------------

word_vectorizer = joblib.load(
    "models/word_tfidf.pkl"
)

char_vectorizer = joblib.load(
    "models/char_tfidf.pkl"
)

model = joblib.load(
    "models/emotion_model_word_char.pkl"
)

print("\nModel and vectorizers loaded successfully.")

# ---------------------------------------------------------
# 5. TRANSFORM UNSEEN DATA
# ---------------------------------------------------------

X_unseen = df["Processed_Sentence"]

X_word = word_vectorizer.transform(
    X_unseen
)

X_char = char_vectorizer.transform(
    X_unseen
)

from scipy.sparse import hstack

X_combined = hstack([
    X_word,
    X_char
])

# ---------------------------------------------------------
# 6. PREDICTION
# ---------------------------------------------------------

y_actual = df["Emotion"]

y_predicted = model.predict(
    X_combined
)

# ---------------------------------------------------------
# 7. ACCURACY
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_actual,
    y_predicted
)

print("\n" + "=" * 70)
print("UNSEEN DATASET ACCURACY")
print("=" * 70)

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)

# ---------------------------------------------------------
# 8. CLASSIFICATION REPORT
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_actual,
        y_predicted,
        zero_division=0
    )
)

# ---------------------------------------------------------
# 9. CONFUSION MATRIX
# ---------------------------------------------------------

labels = sorted(
    df["Emotion"].unique()
)

cm = confusion_matrix(
    y_actual,
    y_predicted,
    labels=labels
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\nLabels:")
print(labels)

print("\nMatrix:")
print(cm)

# ---------------------------------------------------------
# 10. INCORRECT PREDICTIONS
# ---------------------------------------------------------

results = df[
    ["Sentence", "Emotion"]
].copy()

results["Predicted_Emotion"] = y_predicted

results["Correct"] = (
    results["Emotion"]
    ==
    results["Predicted_Emotion"]
)

incorrect = results[
    results["Correct"] == False
]

print("\n" + "=" * 70)
print("INCORRECT PREDICTIONS")
print("=" * 70)

print(
    "\nNumber of incorrect predictions:",
    len(incorrect)
)

if len(incorrect) == 0:

    print("\nNo incorrect predictions.")

else:

    for _, row in incorrect.iterrows():

        print("\nSentence :", row["Sentence"])
        print("Actual   :", row["Emotion"])
        print("Predicted:", row["Predicted_Emotion"])

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)