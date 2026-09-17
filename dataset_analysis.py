import pandas as pd
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("dataset/final_dataset.csv")

df.columns = df.columns.str.strip()

print("=" * 70)
print("EMOTION SENSE - DATASET ANALYSIS")
print("=" * 70)


# --------------------------------------------------
# BASIC INFORMATION
# --------------------------------------------------

print("\nDataset Shape:", df.shape)

print("\nEmotion Distribution:")
print(df["Emotion"].value_counts())


# --------------------------------------------------
# SENTENCES PER EMOTION
# --------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE SENTENCES BY EMOTION")
print("=" * 70)

for emotion in sorted(df["Emotion"].unique()):

    emotion_df = df[df["Emotion"] == emotion]

    print("\n" + "-" * 70)
    print("EMOTION:", emotion)
    print("NUMBER OF SENTENCES:", len(emotion_df))
    print("-" * 70)

    # Show first 10 sentences completely
    for i, sentence in enumerate(
        emotion_df["Sentence"].head(10),
        start=1
    ):
        print(f"{i}. {sentence}")


# --------------------------------------------------
# SENTENCE LENGTH
# --------------------------------------------------

df["Word_Count"] = (
    df["Sentence"]
    .astype(str)
    .apply(lambda x: len(x.split()))
)

print("\n" + "=" * 70)
print("AVERAGE SENTENCE LENGTH")
print("=" * 70)

print(
    df.groupby("Emotion")["Word_Count"]
    .agg(["count", "mean", "min", "max"])
    .round(2)
)


# --------------------------------------------------
# VOCABULARY BY EMOTION
# --------------------------------------------------

print("\n" + "=" * 70)
print("IMPORTANT WORDS BY EMOTION")
print("=" * 70)

for emotion in sorted(df["Emotion"].unique()):

    emotion_text = " ".join(
        df[df["Emotion"] == emotion]["Sentence"]
    )

    words = emotion_text.split()

    # Remove very short words
    words = [
        word for word in words
        if len(word) > 2
    ]

    from collections import Counter

    word_counts = Counter(words)

    print("\n", emotion)
    print(
        word_counts.most_common(15)
    )


# --------------------------------------------------
# COMPLETE SENTENCE LIST
# --------------------------------------------------

print("\n" + "=" * 70)
print("ALL SENTENCES")
print("=" * 70)

for emotion in sorted(df["Emotion"].unique()):

    print("\n\n###", emotion, "###")

    emotion_df = df[
        df["Emotion"] == emotion
    ]

    for i, sentence in enumerate(
        emotion_df["Sentence"],
        start=1
    ):
        print(f"{i}. {sentence}")


print("\n" + "=" * 70)
print("DATASET ANALYSIS COMPLETED")
print("=" * 70)