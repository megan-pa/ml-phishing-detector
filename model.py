import re
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('archive/phishing_email.csv')
df = df.dropna(subset=["text_combined", "label"])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

X = df["text_combined"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

X_train = X_train.apply(clean_text)
X_test = X_test.apply(clean_text)