import joblib
import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import svm

df_text = pd.read_csv('../archive/phishing_email.csv')
df_text= df_text.dropna(subset=["text_combined", "label"])

df_addr = pd.read_csv('../archive/SpamAssasin.csv')
df_addr = df_addr.dropna(subset=["subject", "body", "label"])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

df_text = df_text.rename(columns={"text_combined": "body"})
df_text["subject"] = ""
df_text["sender"] = ""
df_text["body"] = df_text["body"].fillna("")
df_text["label"] = df_text["label"].astype(int)

df_addr["subject"] = df_addr["subject"].fillna("")
df_addr["sender"] = df_addr["sender"].fillna("")
df_addr["body"] = df_addr["body"].fillna("")
df_addr["label"] = df_addr["label"].astype(int)

print(df_text["label"].value_counts())
print(df_addr["label"].value_counts())

df = pd.concat(
    [
        df_text[["subject", "sender", "body", "label"]], 
        df_addr[["subject", "sender", "body", "label"]]
    ], 
    ignore_index=True
)

df["text_combined"] = (
    df["sender"].fillna("").astype(str) + " " +
    df["subject"].fillna("").astype(str) + " " +
    df["body"].fillna("").astype(str)
)

X = df["text_combined"]
y = df["label"].astype(int)

# class imbalance analysis
print("Class distribution: ", y.value_counts())
print("\nClass distribution percentages: ", y.value_counts(normalize=True) * 100)

counts = y.value_counts()
imbalanced_ratio = counts.max() / counts.min()
print(f"\nImbalance Ratio (max/min): {imbalanced_ratio:.2f}x")

# splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

X_train = X_train.apply(clean_text)
X_test = X_test.apply(clean_text)

# logistic regression model pipeline
logistic_pipe = Pipeline(
    [
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ]
)

lr_param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
}

lr_search = GridSearchCV(
    logistic_pipe,
    param_grid=lr_param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=1
)

# linear svm model pipeline
svm_pipe = Pipeline(
    [
        ('vectorizer', TfidfVectorizer()),
        ('classifier', svm.SVC(random_state=42, class_weight='balanced'))
    ]
)

# hyperparameter tuning for SVM
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf']
}

svm_search = RandomizedSearchCV(   
    estimator = svm_pipe,
    param_distributions = param_grid,
    n_iter = 5,
    cv = 3,
    scoring = 'f1',
    random_state = 42,
    n_jobs = -1,
    verbose = 1
)

# function to evaluate accuracy of models
def model_accuracy(fit_model, X_test, y_test, name="Model"):
    y_pred = fit_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{name} Test Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    return accuracy

lr_search.fit(X_train, y_train)
print("Logistic Regression Parameters: ", lr_search.best_params_)
print("Logistic Regression Best F1 Score: ", lr_search.best_score_)

svm_search.fit(X_train, y_train)
print("SVM Parameters: ", svm_search.best_params_)
print("SVM Best F1 Score: ", svm_search.best_score_)

model_accuracy(lr_search, X_test, y_test, name="Logistic Regression (best)")
model_accuracy(svm_search, X_test, y_test, name="SVM (best)")

# saving the best model
if lr_search.best_score_ >= svm_search.best_score_:
    best_model = lr_search.best_estimator_
    best_name = "logistic_regression"
else:
    best_model = svm_search.best_estimator_
    best_name = "svm"

os.makedirs("../artifacts", exist_ok=True)
MODEL_PATH = "../artifacts/best_phishing_model.pkl"
joblib.dump(best_model, MODEL_PATH)
print(f"\nBest model saved to {MODEL_PATH}")
