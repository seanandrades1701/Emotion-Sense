import pandas as pd
import sys
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Enable UTF-8 output for Marathi
sys.stdout.reconfigure(encoding="utf-8")


# --------------------------------------------------
# Load Final Dataset
# --------------------------------------------------

df = pd.read_csv("dataset/final_dataset.csv")

print("=" * 60)
print("EMOTION SENSE - MODEL TRAINING")
print("=" * 60)

print("\nDataset Shape:", df.shape)

print("\nEmotion Distribution:")
print(df["Emotion"].value_counts())


# --------------------------------------------------
# Input and Output
# --------------------------------------------------

X = df["Sentence"]
y = df["Emotion"]


# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records :", len(X_test))


# --------------------------------------------------
# TF-IDF Feature Extraction
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF Feature Extraction Completed")

print("Training Feature Matrix:", X_train_tfidf.shape)
print("Testing Feature Matrix :", X_test_tfidf.shape)


# --------------------------------------------------
# Train Logistic Regression Model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)


print("\nLogistic Regression Model Training Completed")


# --------------------------------------------------
# Prediction
# --------------------------------------------------

y_pred = model.predict(X_test_tfidf)


# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# Save Model and Vectorizer
# --------------------------------------------------

joblib.dump(model, "emotion_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved as: emotion_model.pkl")
print("Vectorizer saved as: tfidf_vectorizer.pkl")


print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)