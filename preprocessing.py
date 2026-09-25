import pandas as pd
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

INPUT_FILE = "dataset/emotionsense_marathi_1200_diverse.csv"
OUTPUT_FILE = "dataset/cleaned_dataset.csv"

print("=" * 70)
print("MARATHI TEXT PREPROCESSING")
print("=" * 70)

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df.columns = df.columns.str.strip()

print("\nDataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nClass Distribution:")
print(df["Emotion"].value_counts())

# --------------------------------------------------
# 2. CHECK MISSING VALUES
# --------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Sentences:", df["Sentence"].duplicated().sum())

# --------------------------------------------------
# 3. TEXT CLEANING
# --------------------------------------------------

def clean_text(text):

    text = str(text)

    # Keep Devanagari, English letters and spaces
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

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["Cleaned_Sentence"] = df["Sentence"].apply(clean_text)

# --------------------------------------------------
# 4. LOAD STOPWORDS
# --------------------------------------------------

with open(
    "stopwords.txt",
    "r",
    encoding="utf-8"
) as file:

    stopwords = set(
        word.strip()
        for word in file
        if word.strip()
    )

# --------------------------------------------------
# 5. TOKENIZATION + STOPWORD REMOVAL
# --------------------------------------------------

def preprocess_text(text):

    tokens = text.split()

    tokens = [
        word
        for word in tokens
        if word not in stopwords
    ]

    return " ".join(tokens)


df["Processed_Sentence"] = (
    df["Cleaned_Sentence"]
    .apply(preprocess_text)
)

# --------------------------------------------------
# 6. DISPLAY PREPROCESSING RESULTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("PREPROCESSING RESULTS")
print("=" * 70)

for i in range(min(10, len(df))):

    print(f"\nRecord {i + 1}")

    print("Original  :", df.loc[i, "Sentence"])

    print("Cleaned   :", df.loc[i, "Cleaned_Sentence"])

    print("Processed :", df.loc[i, "Processed_Sentence"])

    print("Emotion   :", df.loc[i, "Emotion"])

# --------------------------------------------------
# 7. SAVE CLEANED DATASET
# --------------------------------------------------

cleaned_df = df[
    [
        "Sentence",
        "Cleaned_Sentence",
        "Processed_Sentence",
        "Emotion"
    ]
].copy()

cleaned_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)

print("\nCleaned dataset saved successfully.")

print("Location:", OUTPUT_FILE)