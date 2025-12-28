import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm

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

# logisitic regression model pipeline
pipe = Pipeline(
    [
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ]
)

# linear svm model pipeline
svm_pipe = Pipeline(
    [
        ('vectorizer', TfidfVectorizer()),
        ('classifier', svm.SVC(kernel='linear', C=1, random_state=42))
    ]
)

# function to evaluate accuracy of models
def model_accuracy(model, X_train, y_train, X_test, y_test, name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    return accuracy

model_accuracy(pipe, X_train, y_train, X_test, y_test, name="Logistic Regression")
model_accuracy(svm_pipe, X_train, y_train, X_test, y_test, name="Linear SVM")