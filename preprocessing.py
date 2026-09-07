import pandas as pd
import re
import sys

# Enable UTF-8 output for Marathi text
sys.stdout.reconfigure(encoding="utf-8")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/emotionsense_dataset.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# -----------------------------
# Dataset Information
# -----------------------------
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nFirst 5 Records:\n")
print(df.head().to_string())

print("\nDataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = str(text)

    # Remove punctuation while keeping Marathi (Devanagari) characters
    text = re.sub(r"[^\u0900-\u097F\sA-Za-z0-9]", "", text)

    # Remove English and Marathi digits
    text = re.sub(r"[0-9०-९]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Apply Cleaning
df["Cleaned_Sentence"] = df["Sentence"].apply(clean_text)

# -----------------------------
# Load Stopwords
# -----------------------------
with open("stopwords.txt", "r", encoding="utf-8") as file:
    stopwords = set(word.strip() for word in file if word.strip())

# -----------------------------
# Tokenization + Stopword Removal
# -----------------------------
def preprocess_text(text):
    tokens = text.split()

    tokens = [
        word.strip()
        for word in tokens
        if word.strip() and word.strip() not in stopwords
    ]

    return " ".join(tokens)

# Apply preprocessing
df["Processed_Sentence"] = df["Cleaned_Sentence"].apply(preprocess_text)

# -----------------------------
# Display Sample Results
# -----------------------------
print("\n" + "=" * 60)
print("TEXT PREPROCESSING RESULTS")
print("=" * 60)

for i in range(min(5, len(df))):
    print(f"\nRecord {i+1}")
    print("Original  :", df.loc[i, "Sentence"])
    print("Cleaned   :", df.loc[i, "Cleaned_Sentence"])
    print("Processed :", df.loc[i, "Processed_Sentence"])

# -----------------------------
# Save Cleaned Dataset
# -----------------------------
try:
    df.to_csv(
        "dataset/cleaned_dataset.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCleaned dataset saved successfully!")
    print("Location: dataset/cleaned_dataset.csv")

except PermissionError:
    print("\nERROR:")
    print("Please close 'cleaned_dataset.csv' if it is open in Excel or another program.")
    exit()

# -----------------------------
# Create Final Dataset
# -----------------------------
final_df = df[["Processed_Sentence", "Emotion"]].copy()

# Rename column
final_df.rename(
    columns={"Processed_Sentence": "Sentence"},
    inplace=True
)

# -----------------------------
# Save Final Dataset
# -----------------------------
try:
    final_df.to_csv(
        "dataset/final_dataset.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nFinal dataset saved successfully!")
    print("Location: dataset/final_dataset.csv")

except PermissionError:
    print("\nERROR:")
    print("Please close 'final_dataset.csv' if it is open in Excel or another program.")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)