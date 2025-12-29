import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import svm

df = pd.read_csv('archive/phishing_email.csv')
df = df.dropna(subset=["text_combined", "label"])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

X = df["text_combined"]
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

X_train = X_train.apply(clean_text)
X_test = X_test.apply(clean_text)

# logisitic regression model pipeline
logisitic_pipe = Pipeline(
    [
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ]
)

lr_param_grid = {
    "vectorizer__ngram_range": [(1,1), (1,2)],
    "vectorizer__min_df": [1, 3, 5],
    "classifier__C": [0.01, 0.1, 1, 10],
}

lr_search = GridSearchCV(
    logisitic_pipe,
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
        ('classifier', svm.SVC(random_state=42))
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
def model_accuracy(model, X_train, y_train, X_test, y_test, name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    return accuracy

model_accuracy(logisitic_pipe, X_train, y_train, X_test, y_test, name="Logistic Regression")
model_accuracy(svm_search, X_train, y_train, X_test, y_test, name="Linear SVM")
