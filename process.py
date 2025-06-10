# process.py
import pandas as pd
import re, string

class TextPreprocessor:
    """
    Clean raw text: remove URLs, HTML tags, emojis, punctuation, digits.
    """
    def __init__(self):
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.emoji_pattern = re.compile(
            "["  
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE
        )
        self.punct_digit_table = str.maketrans(
            {ch: " " for ch in string.punctuation + string.digits}
        )

    def clean(self, text: str) -> str:
        text = self.url_pattern.sub(" ", str(text))
        text = self.html_pattern.sub(" ", text)
        text = text.translate(self.punct_digit_table)
        text = self.emoji_pattern.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    def __call__(self, series: pd.Series) -> pd.Series:
        return series.fillna("").apply(self.clean)


def main():
    # 1. Load
    df_fake = pd.read_csv("DataSet_Misinfo_FAKE.csv")
    df_true = pd.read_csv("DataSet_Misinfo_TRUE.csv")

    # 2. Gán nhãn: fake=0, true=1
    df_fake["label"] = 0
    df_true["label"] = 1

    # 3. Chọn cột text (ở dataset này gọi là 'text') và label
    df = pd.concat([df_fake[["text", "label"]], df_true[["text", "label"]]], ignore_index=True)

    # 4. Tiền xử lý
    preproc = TextPreprocessor()
    df["text"] = preproc(df["text"])

    # 5. Xuất
    df.to_csv("processed_data.csv", index=False, encoding="utf-8")
    print(f"Processed {len(df)} samples, saved to processed_data.csv")

if __name__ == "__main__":
    main()
