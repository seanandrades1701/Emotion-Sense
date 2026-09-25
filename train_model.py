import pandas as pd
import sys
import joblib
import os

from scipy.sparse import hstack

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

sys.stdout.reconfigure(encoding="utf-8")

print("=" * 70)
print("MARATHI EMOTION CLASSIFICATION")
print("WORD + CHARACTER TF-IDF")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

df = pd.read_csv("dataset/cleaned_dataset.csv")
df.columns = df.columns.str.strip()

print("\nDataset Shape:", df.shape)

print("\nClass Distribution:")
print(df["Emotion"].value_counts())

df = df.dropna(subset=["Processed_Sentence", "Emotion"])

X = df["Processed_Sentence"]
y = df["Emotion"]

# ---------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# ---------------------------------------------------------
# 3. WORD TF-IDF
# ---------------------------------------------------------

print("\nCreating Word TF-IDF...")

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    max_features=8000
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)

print("Word Feature Matrix:", X_train_word.shape)

# ---------------------------------------------------------
# 4. CHARACTER TF-IDF
# ---------------------------------------------------------

print("\nCreating Character TF-IDF...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
    max_features=12000
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)

print("Character Feature Matrix:", X_train_char.shape)

# ---------------------------------------------------------
# 5. COMBINE FEATURES
# ---------------------------------------------------------

print("\nCombining Word + Character features...")

X_train_combined = hstack([
    X_train_word,
    X_train_char
])

X_test_combined = hstack([
    X_test_word,
    X_test_char
])

print("Combined Feature Matrix:", X_train_combined.shape)

# ---------------------------------------------------------
# 6. TRAIN MODEL
# ---------------------------------------------------------

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=3000,
    C=1.0,
    class_weight=None
)

model.fit(X_train_combined, y_train)

print("Training completed.")

# ---------------------------------------------------------
# 7. TRAINING ACCURACY
# ---------------------------------------------------------

train_predictions = model.predict(X_train_combined)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

print(
    "\nTraining Accuracy:",
    round(train_accuracy * 100, 2),
    "%"
)

# ---------------------------------------------------------
# 8. TESTING ACCURACY
# ---------------------------------------------------------

test_predictions = model.predict(X_test_combined)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(
    "Testing Accuracy :",
    round(test_accuracy * 100, 2),
    "%"
)

gap = train_accuracy - test_accuracy

print(
    "\nAccuracy Gap:",
    round(gap * 100, 2),
    "percentage points"
)

if gap > 0.15:
    print("Observation: Possible overfitting.")
elif train_accuracy < 0.65 and test_accuracy < 0.65:
    print("Observation: Possible underfitting.")
else:
    print("Observation: No strong overfitting/underfitting indication.")

# ---------------------------------------------------------
# 9. CLASSIFICATION REPORT
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

# ---------------------------------------------------------
# 10. CONFUSION MATRIX
# ---------------------------------------------------------

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    test_predictions,
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
# 11. 5-FOLD CROSS VALIDATION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("5-FOLD CROSS VALIDATION")
print("=" * 70)

# Combine training features only
cv_features = X_train_combined

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    cv_features,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("\nFold Accuracies:")

for i, score in enumerate(cv_scores, start=1):
    print(
        f"Fold {i}: {score * 100:.2f}%"
    )

print(
    "\nMean CV Accuracy:",
    round(cv_scores.mean() * 100, 2),
    "%"
)

print(
    "CV Standard Deviation:",
    round(cv_scores.std() * 100, 2),
    "%"
)

# ---------------------------------------------------------
# 12. SAVE MODEL + VECTORIZERS
# ---------------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    word_vectorizer,
    "models/word_tfidf.pkl"
)

joblib.dump(
    char_vectorizer,
    "models/char_tfidf.pkl"
)

joblib.dump(
    model,
    "models/emotion_model_word_char.pkl"
)

print("\nModels saved successfully:")

print("models/word_tfidf.pkl")
print("models/char_tfidf.pkl")
print("models/emotion_model_word_char.pkl")

print("\n" + "=" * 70)
print("WORD + CHARACTER MODEL TRAINING COMPLETED")
print("=" * 70)