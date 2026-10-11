
import re
import sys
import joblib


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")

from scipy.sparse import hstack

sys.stdout.reconfigure(encoding="utf-8")

print("=" * 60)
print("       EMOTION SENSE - MARATHI TEXT ANALYZER")
print("=" * 60)

# Load stopwords
with open("stopwords.txt", "r", encoding="utf-8") as file:
    stopwords = set(
        word.strip()
        for word in file
        if word.strip()
    )

# Load trained model and vectorizers
word_vectorizer = joblib.load(
    "models/word_tfidf.pkl"
)

char_vectorizer = joblib.load(
    "models/char_tfidf.pkl"
)

model = joblib.load(
    "models/emotion_model_word_char.pkl"
)

def clean_text(text):
    text = str(text)

    # Preserve Marathi characters and matras
    text = re.sub(
        r"[^\u0900-\u097F\sA-Za-z]",
        "",
        text
    )

    # Remove English and Marathi digits
    text = re.sub(
        r"[0-9०-९]",
        "",
        text
    )

    # Normalize whitespace
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
        word for word in tokens
        if word not in stopwords
    ]

    return " ".join(tokens)

emotion_names = {
    "आनंदी": "Happy",
    "आश्चर्य": "Surprise",
    "दुःखी": "Sad",
    "भीती": "Fear",
    "राग": "Angry",
    "सामान्य": "Neutral"
}

print("\nModel loaded successfully!")

while True:
    print("\n" + "-" * 60)

    sentence = input(
        "Enter a Marathi sentence (or type 'exit' to quit):\n> "
    ).strip()

    if sentence.lower() == "exit":
        print("\nThank you for using Emotion Sense!")
        break

    if not sentence:
        print("Please enter a sentence.")
        continue

    processed = preprocess_text(sentence)

    if not processed:
        print("No usable text remained after preprocessing.")
        continue

    # Extract word and character features
    word_features = word_vectorizer.transform([processed])
    char_features = char_vectorizer.transform([processed])

    combined_features = hstack([
        word_features,
        char_features
    ])

    # Predict emotion
    prediction = model.predict(combined_features)[0]

    print("\nOriginal Sentence :", sentence)
    print("Processed Sentence:", processed)
    print("Predicted Emotion :", prediction)
    print("Emotion in English:", emotion_names.get(
        prediction, "Unknown"
    ))

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(combined_features)[0]
        probability = max(probabilities) * 100

        print(
            "Model Probability :",
            f"{probability:.2f}%"
        )

        print(
            "\nNote: Model probability is not a guarantee "
            "that the prediction is correct."
        )
