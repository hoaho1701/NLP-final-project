# eda.py
import pandas as pd
import matplotlib.pyplot as plt

def run_eda(path="processed_data.csv"):
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("Class distribution:")
    print(df["label"].value_counts())

    # Độ dài câu
    df["length"] = df["text"].apply(lambda t: len(t.split()))
    print("Text length (words):")
    print(df["length"].describe())

    # Plot class balance
    plt.figure(figsize=(4,4))
    df["label"].value_counts().plot.pie(
        labels=["Fake","Real"],
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Class Balance")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    # Histogram độ dài văn bản
    plt.figure(figsize=(6,4))
    plt.hist(df["length"], bins=50)
    plt.xlabel("Number of Words")
    plt.ylabel("Count")
    plt.title("Distribution of Text Length")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_eda()
