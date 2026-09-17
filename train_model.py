import pandas as pd
import sys
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
from matplotlib import rcParams


# ============================================================
# SETTINGS
# ============================================================

sys.stdout.reconfigure(encoding="utf-8")
rcParams["font.family"] = "Nirmala UI"


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/final_dataset.csv")
df.columns = df.columns.str.strip()

print("=" * 65)
print("EMOTION SENSE - OVERFITTING REDUCTION")
print("=" * 65)

print("\nDataset Shape:", df.shape)

print("\nEmotion Distribution:")
print(df["Emotion"].value_counts())


# ============================================================
# INPUT AND OUTPUT
# ============================================================

X = df["Sentence"].astype(str)
y = df["Emotion"].astype(str)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records :", len(X_test))


# ============================================================
# WORD TF-IDF
# ============================================================

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    max_features=1000
)


# ============================================================
# CHARACTER TF-IDF
# ============================================================

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
    max_features=2000
)


# ============================================================
# COMBINE FEATURES
# ============================================================

vectorizer = FeatureUnion([
    ("word_tfidf", word_vectorizer),
    ("char_tfidf", char_vectorizer)
])


# ============================================================
# FEATURE EXTRACTION
# ============================================================

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

print("\n" + "=" * 65)
print("FEATURE EXTRACTION")
print("=" * 65)

print("\nTraining Feature Matrix:",
      X_train_features.shape)

print("Testing Feature Matrix :",
      X_test_features.shape)


# ============================================================
# TEST DIFFERENT REGULARIZATION VALUES
# ============================================================

C_VALUES = [1.0, 0.5, 0.2, 0.1, 0.05]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

print("\n" + "=" * 65)
print("REGULARIZATION EXPERIMENT")
print("=" * 65)

for C in C_VALUES:

    model = LogisticRegression(
        C=C,
        max_iter=2000,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X_train_features,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    mean_score = scores.mean()

    results.append((C, mean_score))

    print(
        f"\nC = {C}"
        f"\nFold Accuracy: "
        f"{[round(score * 100, 2) for score in scores]}"
        f"\nMean CV Accuracy: {mean_score * 100:.2f}%"
    )


# ============================================================
# SELECT BEST C
# ============================================================

best_C, best_cv_score = max(
    results,
    key=lambda x: x[1]
)

print("\n" + "=" * 65)
print("BEST REGULARIZATION")
print("=" * 65)

print("\nBest C:", best_C)

print(
    "Mean Cross-Validation Accuracy:",
    round(best_cv_score * 100, 2),
    "%"
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

final_model = LogisticRegression(
    C=best_C,
    max_iter=2000,
    random_state=42
)

final_model.fit(
    X_train_features,
    y_train
)

print("\nFinal model training completed.")


# ============================================================
# TRAINING ACCURACY
# ============================================================

train_predictions = final_model.predict(
    X_train_features
)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

print(
    "\nTraining Accuracy:",
    round(train_accuracy * 100, 2),
    "%"
)


# ============================================================
# TESTING
# ============================================================

test_predictions = final_model.predict(
    X_test_features
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(
    "Testing Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 65)
print("FINAL CLASSIFICATION REPORT")
print("=" * 65)

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=labels
)

print("\n" + "=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

print(cm)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(
    xticks_rotation=45
)

plt.title(
    "Marathi Emotion Classification - Regularized Model"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix_final.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    final_model,
    "emotion_model.pkl"
)

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 65)
print("FINAL MODEL SAVED")
print("=" * 65)

print("\nModel:")
print("emotion_model.pkl")

print("\nVectorizer:")
print("tfidf_vectorizer.pkl")

print("\nConfusion Matrix:")
print("confusion_matrix_final.png")

print("\n" + "=" * 65)
print("OVERFITTING REDUCTION COMPLETED")
print("=" * 65)